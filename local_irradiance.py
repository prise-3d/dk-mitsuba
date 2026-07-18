import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.spatial import KDTree

def mis_weight(pdf_a, pdf_b):
    return dr.select(pdf_a > 0, pdf_a / (pdf_a + pdf_b), 0)

class SurfaceIrradianceVolume:
    """
    A class that maintains a volume of irradiance estimates at sampled surface points in the scene,
    discretized into directional bins, and provides methods for sampling directions based on learned Q-values.
    """
    def __init__(self, scene, positions, normals, resolution_u=8, resolution_v=8, grid_res=32, q_init_value=1.0, q_init_weight=8.0, grid_k=4):
        """
        Initializes the SurfaceIrradianceVolume with the given scene, sampled positions, normals, and discretization parameters.
        Input:  - scene: The Mitsuba scene object.
                - positions: A list of 3D positions on the scene surfaces where irradiance will be estimated.
                - normals: The corresponding surface normals at the sampled positions.
                - resolution_u, resolution_v: The angular resolution for the directional discretization (number of bins in azimuth and elevation).
                - grid_res: The resolution of the spatial grid for efficient nearest neighbor queries when finding the closest surface point for a given ray intersection.
                - q_init_value, q_init_weight: positive uniform initialization of Q
                (Dahm & Keller 2017, Sec. 3.1), expressed as a pseudo-count prior:
                each bin starts as if it had received q_init_weight visits of
                value q_init_value (radiance units; 0 disables the prior).
                - grid_k: number of candidate probes stored per grid cell,
                among which nearest_point picks the closest normal-compatible one.
        """
        self.positions, self.normals = mi.Point3f(positions), mi.Vector3f(normals)
        self.n_points = dr.width(self.positions)
        self.res_u, self.res_v, self.n_bins_per_point = resolution_u, resolution_v, resolution_u * resolution_v
        total_size = self.n_points * self.n_bins_per_point
        prior = q_init_value * q_init_weight
        self.sum_r, self.sum_g, self.sum_b = dr.full(mi.Float, prior, total_size), dr.full(mi.Float, prior, total_size), dr.full(mi.Float, prior, total_size)
        self.visit_counts = dr.full(mi.Float, q_init_weight, total_size)
        self.bin_cosines = [(mi.Float(i // resolution_u) + 0.5) / resolution_v for i in range(self.n_bins_per_point)]
        # Flat per-bin cosines aligned with the (probe * n_bins + bin) layout
        cos_bins = np.array([(k // resolution_u + 0.5) / resolution_v for k in range(self.n_bins_per_point)], dtype=np.float32)
        self._cos_flat = dr.tile(mi.Float(cos_bins), self.n_points)
        self.grid_res, bbox = grid_res, scene.bbox()
        self.grid_min, self.grid_max = mi.Point3f(bbox.min), mi.Point3f(bbox.max)
        self.grid_size = dr.maximum(self.grid_max - self.grid_min, 1e-4)
        self._build_grid(grid_k)
        # Per-probe sampling CDF and radiance estimates, kept in sync with the
        # Q data by refresh_distributions
        self.cdf = dr.zeros(mi.Float, total_size)
        self.refresh_distributions()

    @classmethod
    def from_scene(cls, scene, n_points, resolution_u=8, resolution_v=8, grid_res=32, q_init_value=1.0, q_init_weight=8.0, grid_k=4):
        """
        Distributes a specified number of probes across the surfaces of the scene (excluding emitters) 
        and constructs a SurfaceIrradianceVolume for RL-guided sampling.
        """
        shapes = [s for s in scene.shapes() if s.emitter() is None]
        # Allocate probes proportionally to surface area (at least one per shape):
        areas = [max(float(s.surface_area()[0]), 1e-8) for s in shapes]
        total_area = sum(areas)
        counts = [max(1, int(round(n_points * a / total_area))) for a in areas]
        # Some shapes (e.g. flat planes) return a scalar normal from sample_position;
        # broadcast to n_per so per-shape concat sizes line up between positions and normals.
        def _bcast(v, n):
            return v if dr.width(v) == n else v + dr.zeros(mi.Float, n)
        px, py, pz, nx, ny, nz = [], [], [], [], [], []
        for i, (s, n_per) in enumerate(zip(shapes, counts)):
            pcg = mi.PCG32(size=n_per, initstate=i)
            ps = s.sample_position(0.0, mi.Point2f(pcg.next_float32(), pcg.next_float32()))
            px.append(_bcast(ps.p.x, n_per)); py.append(_bcast(ps.p.y, n_per)); pz.append(_bcast(ps.p.z, n_per))
            nx.append(_bcast(ps.n.x, n_per)); ny.append(_bcast(ps.n.y, n_per)); nz.append(_bcast(ps.n.z, n_per))

        positions = mi.Point3f(dr.concat(px), dr.concat(py), dr.concat(pz))
        normals = mi.Vector3f(dr.concat(nx), dr.concat(ny), dr.concat(nz))
        
        return cls(scene, positions, normals, resolution_u, resolution_v, grid_res, q_init_value, q_init_weight, grid_k)

    def _build_grid(self, k=4):
        """
        Builds the spatial grid for efficient nearest neighbor queries.
        A KD-tree assigns to every cell its k nearest probes (from the cell
        center)
        """
        res = self.grid_res
        self.grid_k = min(k, self.n_points)

        axis = (np.arange(res) + 0.5) / res
        gmin = np.array(self.grid_min).reshape(3)
        gsize = np.array(self.grid_size).reshape(3)
        # Cell index = ix + iy*res + iz*res^2, so x varies fastest
        zz, yy, xx = np.meshgrid(axis, axis, axis, indexing='ij')
        centers = gmin + gsize * np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        tree = KDTree(np.array(self.positions).T)
        _, idx = tree.query(centers, k=self.grid_k, workers=-1)
        idx = idx.reshape(res**3, self.grid_k)
        self.grid_data = [mi.UInt32(idx[:, j].astype(np.uint32)) for j in range(self.grid_k)]

    def nearest_point(self, p, n, threshold=0.5):
        """
        Finds the nearest surface point index using a spatial grid. O(1) complexity.
        Among the cell's k candidate probes, returns the one closest to p whose
        normal aligns with n. Returns (idx, valid) where valid is True iff at
        least one candidate probe normal aligns with n.
        """
        p_rel = (p - self.grid_min) / self.grid_size
        ix = dr.clip(mi.UInt32(p_rel.x * self.grid_res), 0, self.grid_res - 1)
        iy = dr.clip(mi.UInt32(p_rel.y * self.grid_res), 0, self.grid_res - 1)
        iz = dr.clip(mi.UInt32(p_rel.z * self.grid_res), 0, self.grid_res - 1)
        idx = ix + iy * self.grid_res + iz * (self.grid_res**2)
        best_idx = dr.zeros(mi.UInt32, dr.width(idx))
        best_d2 = dr.full(mi.Float, 1e30, dr.width(idx))
        valid = dr.full(mi.Bool, False, dr.width(idx))
        for j in range(self.grid_k):
            cand = dr.gather(mi.UInt32, self.grid_data[j], idx)
            cand_p = dr.gather(mi.Point3f, self.positions, cand)
            cand_n = dr.gather(mi.Vector3f, self.normals, cand)
            ok = dr.dot(n, cand_n) > threshold
            d2 = dr.squared_norm(p - cand_p)
            better = ok & (d2 < best_d2)
            best_idx = dr.select(better, cand, best_idx)
            best_d2 = dr.select(better, d2, best_d2)
            valid = valid | ok
        return best_idx, valid

    def update(self, spatial_indices, frame_n, directions, rewards, active):
        """
        Updates the Q-values for the given spatial indices, directions, and rewards based on the observed samples.
        Input:  - spatial_indices: The indices of the surface points corresponding to the samples.
                - directions: The world-space direction vectors of the samples.
                - rewards: The observed rewards (radiance) for the samples, as a Color3f.
                - active: A boolean mask indicating which samples are active and should be updated.
        """
        w_l = mi.Frame3f(frame_n).to_local(directions)
        active = active & (w_l.z > 0)
        phi = dr.atan2(w_l.y, w_l.x)
        u_idx = dr.minimum(mi.UInt32((phi / (2*dr.pi) + dr.select(phi < 0, 1.0, 0.0)) * self.res_u), self.res_u - 1)
        v_idx = dr.minimum(mi.UInt32(dr.clip(w_l.z, 0.0, 1.0) * self.res_v), self.res_v - 1)
        flat_idx = spatial_indices * self.n_bins_per_point + (v_idx * self.res_u + u_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, self.sum_r, rewards.x, flat_idx, active)
        dr.scatter_reduce(dr.ReduceOp.Add, self.sum_g, rewards.y, flat_idx, active)
        dr.scatter_reduce(dr.ReduceOp.Add, self.sum_b, rewards.z, flat_idx, active)
        dr.scatter_reduce(dr.ReduceOp.Add, self.visit_counts, 1.0, flat_idx, active)

    def get_q_data(self, spatial_indices):
        """
        Retrieves the Q-values for the given spatial indices.
        """
        all_q = []
        for i in range(self.n_bins_per_point):
            flat_idx = spatial_indices * self.n_bins_per_point + i
            count = dr.maximum(dr.gather(mi.Float, self.visit_counts, flat_idx), 1.0)
            all_q.append(mi.Color3f(dr.gather(mi.Float, self.sum_r, flat_idx) / count, dr.gather(mi.Float, self.sum_g, flat_idx) / count, dr.gather(mi.Float, self.sum_b, flat_idx) / count))
        return all_q

    def get_q_sum(self, spatial_indices):
        """
        Computes the sum of Q-values across all bins for the given spatial indices, weighted by the cosine of the bin directions.
        """
        all_q, res = self.get_q_data(spatial_indices), mi.Color3f(0.0)
        for i, q in enumerate(all_q): res += q * self.bin_cosines[i]
        return mi.luminance(res)

    def refresh_distributions(self, threshold=0.01, relative=True):
        """
        Rebuilds the per-probe sampling distributions and radiance estimates
        from the current Q data. All the O(n_bins) work happens here at probe
        width (n_points lanes), so per-path queries in sample_direction /
        pdf_direction / compute_radiance_estimate reduce to a few gathers.
        The bin weights are proportional to max(Q, eps) * cos with a positive
        clamp for ergodicity (Dahm & Keller 2017, Section 3.1).
        With relative=True (default), eps is threshold * max_bin(Q) per probe:
        an absolute clamp either flattens the learned distribution where Q is
        small (e.g. indirectly lit regions) or lets never-revisited bins freeze
        at Q=0, while a relative floor scales with the local signal and falls
        back to cosine sampling where nothing is learned yet.
        With relative=False, threshold is the absolute clamp value.
        """
        B = self.n_bins_per_point
        d_omega = (2 * dr.pi) / B
        count = dr.maximum(self.visit_counts, 1.0)
        q_r, q_g, q_b = self.sum_r / count, self.sum_g / count, self.sum_b / count
        lum = mi.luminance(mi.Color3f(q_r, q_g, q_b))

        # Per-probe radiance estimate (Sarsa target): one block reduction per channel
        scale = d_omega / dr.pi
        self.rad_r = dr.block_reduce(dr.ReduceOp.Add, q_r * self._cos_flat, B) * scale
        self.rad_g = dr.block_reduce(dr.ReduceOp.Add, q_g * self._cos_flat, B) * scale
        self.rad_b = dr.block_reduce(dr.ReduceOp.Add, q_b * self._cos_flat, B) * scale

        if relative:
            q_max = dr.block_reduce(dr.ReduceOp.Max, lum, B)
            eps = dr.repeat(dr.maximum(q_max * threshold, 1e-8), B)
        else:
            eps = mi.Float(threshold)
        w = dr.maximum(lum, eps) * self._cos_flat
        total = dr.repeat(dr.block_reduce(dr.ReduceOp.Add, w, B), B)
        # Per-probe inclusive CDF in one segmented prefix sum
        self.cdf = dr.block_prefix_reduce(dr.ReduceOp.Add, w / total, block_size=B, exclusive=False)

    def _cdf_interval(self, spatial_indices, bin_idx):
        """Returns (prev_c, cur_c), the CDF values bracketing bin_idx."""
        base = spatial_indices * self.n_bins_per_point
        cur_c = dr.gather(mi.Float, self.cdf, base + bin_idx)
        prev_c = dr.select(bin_idx > 0,
                            dr.gather(mi.Float, self.cdf, base + dr.maximum(bin_idx, 1) - 1),
                            0.0)
        return prev_c, cur_c

    def _map_to_world_direction(self, spatial_indices, frame_n, bin_idx, u_l, sample_y):
        """
        Maps the selected bin and continuous samples to a world-space direction vector.
        Reference: Dahm & Keller (2017), Section 3.1 - Spatial and Directional Discretization.
        """
        phi = (mi.Float(bin_idx % self.res_u) + u_l) * (2 * dr.pi / self.res_u)
        cos_theta = dr.clip((mi.Float(bin_idx // self.res_u) + sample_y) / self.res_v, 0.0, 1.0)
        sin_theta = dr.safe_sqrt(1.0 - cos_theta * cos_theta)

        local_dir = mi.Vector3f(sin_theta * dr.cos(phi), sin_theta * dr.sin(phi), cos_theta)
        return mi.Frame3f(frame_n).to_world(local_dir)

    def sample_direction(self, spatial_indices, frame_n, sample):
        """Samples a direction based on the learned Q-values and returns the corresponding PDF.
        Inverts the per-probe CDF built by refresh_distributions with a binary
        search: O(log n_bins) gathers per path instead of per-path weight
        recomputation.
        Input:  - spatial_indices: The indices of the surface points for which to sample directions.
                - sample: A 2D sample (sample.x, sample.y) in the range [0, 1] used for sampling the bin and the local offset.
        Output: - direction: The sampled world-space direction vector.
                - pdf: The probability density function value for the sampled direction.
        """
        base = spatial_indices * self.n_bins_per_point
        lo = dr.zeros(mi.UInt32, dr.width(spatial_indices))
        hi = dr.full(mi.UInt32, self.n_bins_per_point - 1, dr.width(spatial_indices))
        # Smallest bin whose (inclusive) CDF value reaches sample.x
        for _ in range(max(1, (self.n_bins_per_point - 1).bit_length())):
            mid = (lo + hi) >> 1
            go_up = sample.x > dr.gather(mi.Float, self.cdf, base + mid)
            lo = dr.select(go_up, mid + 1, lo)
            hi = dr.select(go_up, hi, mid)
        bin_idx = lo

        prev_c, cur_c = self._cdf_interval(spatial_indices, bin_idx)
        u_local = dr.clip((sample.x - prev_c) / dr.maximum(cur_c - prev_c, 1e-7), 0.0, 1.0)
        direction = self._map_to_world_direction(spatial_indices, frame_n, bin_idx, u_local, sample.y)
        return direction, (cur_c - prev_c) * (self.n_bins_per_point / (2 * dr.pi))

    def pdf_direction(self, spatial_indices, frame_n, directions):
        """Computes the PDF for given directions based on the learned Q-values.
        Input:  - spatial_indices: The indices of the surface points for which to compute the PDF.
                - directions: The world-space direction vectors for which to compute the PDF.
        Output: - pdf: The probability density function values for the given directions."""
        # Determine which bin the given directions fall into
        w_l = mi.Frame3f(frame_n).to_local(directions)
        phi = dr.atan2(w_l.y, w_l.x)
        u_idx = dr.minimum(mi.UInt32((phi / (2*dr.pi) + dr.select(phi < 0, 1.0, 0.0)) * self.res_u), self.res_u - 1)
        v_idx = dr.minimum(mi.UInt32(dr.clip(w_l.z, 0.0, 1.0) * self.res_v), self.res_v - 1)
        bin_idx = v_idx * self.res_u + u_idx

        prev_c, cur_c = self._cdf_interval(spatial_indices, bin_idx)
        return dr.select(w_l.z > 0, (cur_c - prev_c) * (self.n_bins_per_point / (2 * dr.pi)), 0.0)

    def compute_radiance_estimate(self, spatial_indices):
        """Per-probe radiance estimate, precomputed by refresh_distributions."""
        return mi.Color3f(dr.gather(mi.Float, self.rad_r, spatial_indices),
                          dr.gather(mi.Float, self.rad_g, spatial_indices),
                          dr.gather(mi.Float, self.rad_b, spatial_indices))

    def get_total_visits(self, spatial_indices):
        total = dr.zeros(mi.Float, dr.width(spatial_indices))
        for i in range(self.n_bins_per_point): total += dr.gather(mi.Float, self.visit_counts, spatial_indices * self.n_bins_per_point + i)
        return total

    def get_stats(self):
        v = self.visit_counts
        return {"total_visits": dr.sum(v)[0], "max_q": dr.max(self.sum_r/dr.maximum(v,1.0))[0], "mean_q": dr.sum(self.sum_r)[0]/dr.maximum(dr.sum(v)[0],1.0)}

    def save(self, path):
        """Saves the sampled positions and normals to a PLY file for visualization."""
        pos, norm = np.array(self.positions), np.array(self.normals)
        with open(path, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {self.n_points}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write("end_header\n")
            for i in range(self.n_points):
                f.write(f"{pos[0, i]} {pos[1, i]} {pos[2, i]} {norm[0, i]} {norm[1, i]} {norm[2, i]}\n")

    def save_hemi(self, path, radius=10.0):
        """Saves the hemisphere visualization of the learned Q-values for each point.
        uses the same PLY format but creates quads for each bin direction colored by the Q-values."""
        res_u, res_v = self.res_u, self.res_v
        n_bins, n_points = self.n_bins_per_point, self.n_points

        with open(path, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n_points * n_bins * 4}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float r\nproperty float g\nproperty float b\n")
            f.write(f"element face {n_points * n_bins}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")

            # Write vertices for each bin direction, colored by the Q-values
            # Represent every vertex needed later
            total      = n_points * n_bins * 4
            lane       = dr.arange(mi.UInt32, total)
            probe_idx  = lane // (n_bins * 4)
            bin_idx    = (lane // 4) % n_bins
            corner_idx = lane % 4

            # compute corner offsets w/ DrJit
            du = dr.select((corner_idx == 1) | (corner_idx == 2), 1, 0)                                                                                                            
            dv = dr.select(corner_idx >= 2, 1, 0)

            # build the local direction in w/ DrJit
            u_idx     = bin_idx %  res_u                                                                                                                                               
            v_idx     = bin_idx // res_u                                                                                                                                               
            phi       = mi.Float(u_idx + du) * (2 * dr.pi / res_u)
            cos_theta = dr.clip(mi.Float(v_idx + dv) / res_v, 0.0, 1.0)                                                                                                            
            sin_theta = dr.safe_sqrt(1.0 - cos_theta * cos_theta)                                                                                                                  
            local_dir = mi.Vector3f(sin_theta * dr.cos(phi),                                                                                                                       
                                    sin_theta * dr.sin(phi),                                                                                                                       
                                    cos_theta)
            
            # gather positions, normals and colors per lane
            p  = dr.gather(mi.Point3f,  self.positions, probe_idx)                                                                                                                 
            n  = dr.gather(mi.Vector3f, self.normals,   probe_idx)
                                                                                                                                                                                    
            flat_pb = probe_idx * n_bins + bin_idx        # bin slot in the SoA Q arrays                                                                                           
            counts  = dr.maximum(dr.gather(mi.Float, self.visit_counts, flat_pb), 1.0)                                                                                             
            r       = dr.gather(mi.Float, self.sum_r, flat_pb) / counts                                                                                                                  
            g       = dr.gather(mi.Float, self.sum_g, flat_pb) / counts                                                                                                                  
            b       = dr.gather(mi.Float, self.sum_b, flat_pb) / counts

            # compute the frame
            v_pos = p + mi.Frame3f(n).to_world(local_dir) * radius

            # sync to numpy
            xyz   = np.asarray(v_pos).T          # shape (total, 3)                                                                                                                  
            rs    = np.asarray(r)                # shape (total,)
            gs    = np.asarray(g)                                                                                                                                                    
            bs    = np.asarray(b)
            verts = np.column_stack([xyz, rs, gs, bs])   # shape (total, 6)

            # stream vertices
            np.savetxt(f, verts, fmt="%g %g %g %g %g %g")

            # ~ Same thing, for the faces
            n_faces     = n_points * n_bins # 1 row per (probe, bin)
            face_lane   = dr.arange(mi.UInt32, n_faces)
            flat_pb_f   = face_lane                                                                            
            counts_f    = dr.maximum(dr.gather(mi.Float, self.visit_counts, flat_pb_f), 1.0)                                                                                       
            rf = dr.clip(dr.gather(mi.Float, self.sum_r, flat_pb_f) / counts_f, 0.0, 1.0)                                                                                          
            gf = dr.clip(dr.gather(mi.Float, self.sum_g, flat_pb_f) / counts_f, 0.0, 1.0)                                                                                          
            bf = dr.clip(dr.gather(mi.Float, self.sum_b, flat_pb_f) / counts_f, 0.0, 1.0)
            
            R     = (np.asarray(rf) * 255).astype(np.int32)                                                                                                                            
            G     = (np.asarray(gf) * 255).astype(np.int32)                                                                                                                            
            B     = (np.asarray(bf) * 255).astype(np.int32)
            v0    = np.arange(n_faces, dtype=np.int32) * 4                                                                                                                         
            fours = np.full(n_faces, 4, dtype=np.int32)
            faces = np.column_stack([fours, v0, v0 + 1, v0 + 2, v0 + 3, R, G, B])                                                                                                        
            np.savetxt(f, faces, fmt="%d")

class RLIntegrator(mi.SamplingIntegrator):
    """
    A custom integrator that implements reinforcement learning-based guiding 
    using a SurfaceIrradianceVolume.
    """

    def __init__(self, props=mi.Properties()):
        """Initializes the RLIntegrator with the given properties.
        Input: - props: A Mitsuba Properties object containing configuration parameters for the integrator.
        Expected properties include:
        - n_probes: The number of probes to distribute across the scene surfaces for learning 
        - enable_guiding: A boolean flag to enable or disable RL-guided sampling.
        - update_q: A boolean flag to update the Q-values during sampling.
        - resolution_u, resolution_v: The angular resolution for the directional discretization in the SurfaceIrradianceVolume.
        - grid_res: The resolution of the spatial grid for nearest neighbor queries in the SurfaceIrradianceVolume.
        """
        super().__init__(props)
        self.n_probes, self.enable_guiding, self.update_q = props.get('n_probes', 1000), props.get('enable_guiding', True), props.get('update_q', True)
        self.resolution_u, self.resolution_v = props.get('resolution_u', 8), props.get('resolution_v', 8)
        self.grid_res = props.get('grid_res', 32)
        self.max_depth = props.get('max_depth', 8)
        self.q_init_value = props.get('q_init_value', 1.0)
        self.q_init_weight = props.get('q_init_weight', 8.0)
        # When to rebuild the per-probe sampling distributions from Q:
        # 'frame' once per frame (Dahm & Keller Sec. 3.1), 'bounce' at every
        # bounce (fresher distributions, higher fixed cost per frame).
        self.refresh_mode = props.get('refresh', 'frame')
        if self.refresh_mode not in ('frame', 'bounce'):
            raise ValueError(f"refresh must be 'frame' or 'bounce', got '{self.refresh_mode}'")
        self.grid_k = props.get('grid_k', 4)
        self.volume = None
        self.next_event_estimation = True

    def sample(self, scene, sampler, ray, medium, active, update_q=True):
        """
        Performs path tracing with optional RL-guided sampling and Q-value updates.
        """
        if self.enable_guiding and self.volume is None:
            self.volume = SurfaceIrradianceVolume.from_scene(
                scene, self.n_probes,
                self.resolution_u, self.resolution_v,
                self.grid_res,
                self.q_init_value, self.q_init_weight,
                self.grid_k
            )
            
        if self.enable_guiding and self.refresh_mode == 'frame':
            # per-probe sampling distributions from the Q data
            # accumulated so far, once per frame, as in Dahm & Keller
            # Sec. 3.1 ("every accumulated frame")
            self.volume.refresh_distributions()

        throughput, result = mi.Spectrum(1.0), mi.Spectrum(0.0)

        prev_idx, prev_dir, has_prev = dr.zeros(mi.UInt32, dr.width(active)),mi.Vector3f(0.0), dr.full(mi.Bool, False, dr.width(active))

        # solid angle pdf used to generate a ray
        prev_pdf = mi.Float(1.0)

        # was the previous sample a specular reflection?
        prev_delta = mi.Bool(True)
        
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_frame_n = mi.Vector3f(0, 0, 1)
        prev_valid = mi.Bool(False)

        for _ in range(self.max_depth):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            
            nee_contrib_val = mi.Spectrum(0.0)
            bsdf = si.bsdf(ray)
            ctx = mi.BSDFContext()

            # probe index is queried only once and weights are pre-computed
            if self.enable_guiding:
                curr_idx, curr_valid = self.volume.nearest_point(si.p, si.sh_frame.n)
                # Pure specular reflections support:
                # BSDF sampling fallback
                can_guide = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                alpha = dr.select(curr_valid & can_guide, 1.0, 0.0)
                if self.refresh_mode == 'bounce':
                    # Different refreshing policy: distributions also pick up
                    # the Q updates deposited at the previous bounce
                    # of this frame.
                    self.volume.refresh_distributions()
            else:
                curr_idx = dr.zeros(mi.UInt32, dr.width(active))
                curr_valid = mi.Bool(False)
                alpha = mi.Float(0.0)

            # Next Event Estimation (NEE)
            if self.next_event_estimation:
                # sampler_emitter_direction already mask inactive lanes, so it is ok -- and much faster -- to not check agains dr.any(active)
                emitter_sample, emitter_weight = scene.sample_emitter_direction(si, sampler.next_2d(active), True, active=active)
                active_nee = active & (mi.luminance(emitter_weight) > 0)
                wo_nee = si.to_local(emitter_sample.d)
                shadow_ray = si.spawn_ray_to(emitter_sample.p)
                # same here: active_nee already masks
                occluded = scene.ray_test(shadow_ray, active_nee)
                bsdf_pdf_at_em = bsdf.pdf(ctx, si, wo_nee, active_nee)
                if self.enable_guiding:
                    pdf_rl_at_em = self.volume.pdf_direction(curr_idx, si.sh_frame.n, emitter_sample.d)
                    pt_pdf_at_em = alpha * pdf_rl_at_em + (1.0 - alpha) * bsdf_pdf_at_em
                else:
                    pt_pdf_at_em = bsdf_pdf_at_em
                pt_pdf_at_em = dr.select(emitter_sample.delta, 0.0, pt_pdf_at_em)
                w_nee = mis_weight(emitter_sample.pdf, pt_pdf_at_em)
                # bsdf.eval already includes the cosine term
                nee_contrib_val = dr.select(active_nee & ~occluded,
                                            emitter_weight * bsdf.eval(ctx, si, wo_nee, active_nee),
                                            0.0)
                result += throughput * w_nee * nee_contrib_val

            # update the Q-values based on the previous action
            if self.update_q and self.enable_guiding:
                # on the first bounce, has_prev is all False, propagated to volume.update through active_up
                # Emitter hits seed the Q table with L_dir and need no probe at the
                # hit point (emitters carry no probes, so requiring curr_valid there
                # would discard the seed rewards); only the indirect estimate, which
                # reads Q at the hit point, requires curr_valid.
                active_up = has_prev & si.is_valid() & prev_valid
                emitter = si.emitter(scene, active_up)
                is_em = emitter != None
                L_dir = dr.select(is_em, emitter.eval(si, active_up), 0.0)
                L_ind = dr.select(~is_em & curr_valid, bsdf.eval_diffuse_reflectance(si) * self.volume.compute_radiance_estimate(curr_idx), 0.0)
                # Le reward est la radiance sortante de si (émise + réfléchie)
                self.volume.update(prev_idx, prev_frame_n, prev_dir, L_dir + L_ind, active_up & (is_em | curr_valid))

            emitter_hit = si.emitter(scene, active)
            active_em_hit = active & (emitter_hit != None)
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            em_pdf = scene.pdf_emitter_direction(prev_si, ds, active_em_hit & ~prev_delta)
            w_bsdf = mis_weight(prev_pdf, em_pdf)

            result += dr.select(active_em_hit,
                                throughput * w_bsdf * emitter_hit.eval(si, active),
                                mi.Spectrum(0.0)
                                )

            # --- Guided Sampling with Robust MIS ---
            if self.enable_guiding:
                # Tirage de la direction (RL ou BSDF)
                wo_rl, pdf_rl = self.volume.sample_direction(curr_idx, si.sh_frame.n, sampler.next_2d(active))
                bs_s, bs_w = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)

                direction = dr.select(sampler.next_1d(active) < alpha, wo_rl, si.to_world(bs_s.wo))
                wo_local = si.to_local(direction)

                # alpha is boolean, so no need to mix pdfs with it
                pdf_mix = pdf_rl

                # throughput update (f * cos / pdf_mix) -- cosine term already included
                weight = dr.select(pdf_mix > 1e-7, bsdf.eval(ctx, si, wo_local, active) / pdf_mix, 0.0)
                throughput *= dr.select(alpha > 0, weight, bs_w)

                prev_idx, prev_dir, has_prev = curr_idx, direction, active & (dr.any(throughput > 0.0))
                bsdf_delta = mi.has_flag(bs_s.sampled_type, mi.BSDFFlags.Delta)
                prev_pdf = dr.select(alpha > 0, pdf_mix, bs_s.pdf)
                prev_delta = dr.select(alpha > 0, mi.Bool(False), bsdf_delta)
                prev_si = si
                prev_frame_n = si.sh_frame.n
                prev_valid = curr_valid
            else:
                bs_s, bs_w = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)
                direction, throughput, has_prev = si.to_world(bs_s.wo), throughput * bs_w, dr.full(mi.Bool, False, dr.width(active))
                prev_pdf = bs_s.pdf
                prev_delta = mi.has_flag(bs_s.sampled_type, mi.BSDFFlags.Delta)
                prev_frame_n = si.sh_frame.n
                prev_si = si
                prev_valid = curr_valid

            ray = si.spawn_ray(direction)
            active = active & (mi.luminance(throughput) > 0.0)
        return result, active, []
    
    def save_hemi_q_values(self, path):
        """Saves the hemisphere visualization of the learned Q-values for each point."""
        if self.volume is not None:
            self.volume.save_hemi(path)

"""
Registers the RLIntegrator with Mitsuba, allowing it to be used in scene descriptions and rendering.
"""
mi.register_integrator("rl_integrator", lambda props: RLIntegrator(props))
