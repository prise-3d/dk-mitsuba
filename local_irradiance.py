import drjit as dr
import mitsuba as mi
import numpy as np

class SurfaceIrradianceVolume:
    def __init__(self, scene, positions, normals, resolution_u=8, resolution_v=8, grid_res=16):
        self.positions, self.normals = mi.Point3f(positions), mi.Vector3f(normals)
        self.n_points = dr.width(self.positions)
        self.res_u, self.res_v, self.n_bins_per_point = resolution_u, resolution_v, resolution_u * resolution_v
        total_size = self.n_points * self.n_bins_per_point
        self.sum_r, self.sum_g, self.sum_b, self.visit_counts = dr.zeros(mi.Float, total_size), dr.zeros(mi.Float, total_size), dr.zeros(mi.Float, total_size), dr.zeros(mi.Float, total_size)
        self.bin_cosines = [(mi.Float(i // resolution_u) + 0.5) / resolution_v for i in range(self.n_bins_per_point)]
        self.grid_res, bbox = grid_res, scene.bbox()
        self.grid_min, self.grid_max = mi.Point3f(bbox.min), mi.Point3f(bbox.max)
        self.grid_size = dr.maximum(self.grid_max - self.grid_min, 1e-4)
        self._build_grid()

    def _build_grid(self):
        res = self.grid_res
        idx = dr.arange(mi.UInt32, res**3)
        p = self.grid_min + self.grid_size * mi.Vector3f((mi.Float(idx % res) + 0.5) / res, (mi.Float((idx // res) % res) + 0.5) / res, (mi.Float(idx // (res * res)) + 0.5) / res)
        best_indices, min_dist2 = dr.zeros(mi.UInt32, dr.width(p)), dr.full(mi.Float, 1e30, dr.width(p))
        for i in range(self.n_points):
            d2 = dr.squared_norm(p - dr.gather(mi.Point3f, self.positions, i))
            is_closer = d2 < min_dist2
            min_dist2, best_indices = dr.select(is_closer, d2, min_dist2), dr.select(is_closer, mi.UInt32(i), best_indices)
        self.grid_data = best_indices

    def nearest_point(self, p, n):
        """
        Finds the nearest surface point index for a given 3D position p and normal n.
        Restricts the search to points with a normal in the same direction (dot product > 0).
        """
        p_rel = (p - self.grid_min) / self.grid_size
        ix = dr.clip(mi.UInt32(p_rel.x * self.grid_res), 0, self.grid_res - 1)
        iy = dr.clip(mi.UInt32(p_rel.y * self.grid_res), 0, self.grid_res - 1)
        iz = dr.clip(mi.UInt32(p_rel.z * self.grid_res), 0, self.grid_res - 1)
        idx = ix + iy * self.grid_res + iz * (self.grid_res**2)
        grid_idx = dr.gather(mi.UInt32, self.grid_data, idx)

        # Verify alignment. If misaligned, we fallback to a search over all points.
        is_aligned = dr.dot(n, dr.gather(mi.Vector3f, self.normals, grid_idx)) > 0

        # Fallback: find the nearest point among those with positive normal alignment.
        min_dist2, fallback_idx = dr.full(mi.Float, 1e30, dr.width(p)), dr.zeros(mi.UInt32, dr.width(p))
        for i in range(self.n_points):
            n_i = dr.gather(mi.Vector3f, self.normals, i)
            p_i = dr.gather(mi.Point3f, self.positions, i)
            valid = dr.dot(n, n_i) > 0
            dist2 = dr.select(valid, dr.squared_norm(p - p_i), 1e31)
            closer = dist2 < min_dist2
            min_dist2, fallback_idx = dr.select(closer, dist2, min_dist2), dr.select(closer, mi.UInt32(i), fallback_idx)

        return dr.select(is_aligned, grid_idx, fallback_idx)

    def update(self, spatial_indices, directions, rewards, active):
        n = dr.gather(mi.Vector3f, self.normals, spatial_indices)
        w_l = mi.Frame3f(n).to_local(directions)
        phi = dr.atan2(w_l.y, w_l.x)
        u_idx = dr.minimum(mi.UInt32((phi / (2*dr.pi) + dr.select(phi < 0, 1.0, 0.0)) * self.res_u), self.res_u - 1)
        v_idx = dr.minimum(mi.UInt32(dr.clip(w_l.z, 0.0, 1.0) * self.res_v), self.res_v - 1)
        flat_idx = spatial_indices * self.n_bins_per_point + (v_idx * self.res_u + u_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, self.sum_r, rewards.x, flat_idx, active)
        dr.scatter_reduce(dr.ReduceOp.Add, self.sum_g, rewards.y, flat_idx, active)
        dr.scatter_reduce(dr.ReduceOp.Add, self.sum_b, rewards.z, flat_idx, active)
        dr.scatter_reduce(dr.ReduceOp.Add, self.visit_counts, 1.0, flat_idx, active)

    def get_q_data(self, spatial_indices):
        all_q = []
        for i in range(self.n_bins_per_point):
            flat_idx = spatial_indices * self.n_bins_per_point + i
            count = dr.maximum(dr.gather(mi.Float, self.visit_counts, flat_idx), 1.0)
            all_q.append(mi.Color3f(dr.gather(mi.Float, self.sum_r, flat_idx) / count, dr.gather(mi.Float, self.sum_g, flat_idx) / count, dr.gather(mi.Float, self.sum_b, flat_idx) / count))
        return all_q

    def get_q_sum(self, spatial_indices):
        all_q, res = self.get_q_data(spatial_indices), mi.Color3f(0.0)
        for i, q in enumerate(all_q): res += q * self.bin_cosines[i]
        return mi.luminance(res)

    def _compute_weights(self, spatial_indices, epsilon=0.1):
        """
        Computes the probability weights for each bin using an epsilon-greedy strategy.
        Reference: Dahm & Keller (2017), Eq. (5) - Probability distribution P_i
        """
        all_q = self.get_q_data(spatial_indices)
        q_cos = [mi.luminance(q) * self.bin_cosines[i] for i, q in enumerate(all_q)]
        
        q_sum = dr.zeros(mi.Float, dr.width(spatial_indices))
        for val in q_cos: q_sum += val

        # Mix learned distribution with uniform distribution for exploration
        weights = [
            dr.select(q_sum > 1e-6, 
                      (1.0 - epsilon) * q_c / q_sum + epsilon / self.n_bins_per_point, 
                      1.0 / self.n_bins_per_point) 
            for q_c in q_cos
        ]
        return weights

    def _sample_bin_discrete(self, weights, sample_x):
        """
        Selects a bin index and calculates the local offset using Inverse Transform Sampling.
        Reference: Dahm & Keller (2017), Section 3.1 - Sampling the piece-wise constant PDF.
        """
        cum_w, curr = [], dr.zeros(mi.Float, dr.width(sample_x))
        for w in weights: curr += w; cum_w.append(curr)

        bin_idx = dr.zeros(mi.UInt32, dr.width(sample_x))
        for i in range(self.n_bins_per_point - 1):
            bin_idx = dr.select(sample_x > cum_w[i], mi.UInt32(i + 1), bin_idx)

        prev_c = dr.zeros(mi.Float, dr.width(sample_x))
        for i in range(self.n_bins_per_point): prev_c = dr.select(bin_idx == i, (cum_w[i-1] if i > 0 else 0.0), prev_c)
        
        # Calculate relative position within the bin for continuous sampling
        bin_width = dr.maximum(dr.gather(mi.Float, dr.concat(cum_w), bin_idx) - prev_c, 1e-7)
        u_local = dr.clip((sample_x - prev_c) / bin_width, 0.0, 1.0)
        
        return bin_idx, u_local

    def _map_to_world_direction(self, spatial_indices, bin_idx, u_l, sample_y):
        """
        Maps the selected bin and continuous samples to a world-space direction vector.
        Reference: Dahm & Keller (2017), Section 3.1 - Spatial and Directional Discretization.
        """
        phi = (mi.Float(bin_idx % self.res_u) + u_l) * (2 * dr.pi / self.res_u)
        cos_theta = dr.clip((mi.Float(bin_idx // self.res_u) + sample_y) / self.res_v, 0.0, 1.0)
        sin_theta = dr.safe_sqrt(1.0 - cos_theta * cos_theta)
        
        local_dir = mi.Vector3f(sin_theta * dr.cos(phi), sin_theta * dr.sin(phi), cos_theta)
        return mi.Frame3f(dr.gather(mi.Vector3f, self.normals, spatial_indices)).to_world(local_dir)

    def sample_direction(self, spatial_indices, sample):
        """Samples a direction based on the learned Q-values and returns the corresponding PDF.
        Input: - spatial_indices: The indices of the surface points for which to sample directions.
               - sample: A 2D sample (sample.x, sample.y) in the range [0, 1] used for sampling the bin and the local offset.
        Output: - direction: The sampled world-space direction vector.
                - pdf: The probability density function value for the sampled direction.
        """
        weights = self._compute_weights(spatial_indices)
        bin_idx, u_l = self._sample_bin_discrete(weights, sample.x)
        direction = self._map_to_world_direction(spatial_indices, bin_idx, u_l, sample.y)

        pdf = dr.zeros(mi.Float, dr.width(spatial_indices))
        for i in range(self.n_bins_per_point): pdf = dr.select(bin_idx == i, weights[i], pdf)
        return direction, pdf * (self.n_bins_per_point / (2 * dr.pi))

    def pdf_direction(self, spatial_indices, directions):
        """Computes the PDF for given directions based on the learned Q-values.
        Input: - spatial_indices: The indices of the surface points for which to compute the PDF.
               - directions: The world-space direction vectors for which to compute the PDF.
        Output: - pdf: The probability density function values for the given directions."""
        weights = self._compute_weights(spatial_indices)

        # Determine which bin the given directions fall into
        n = dr.gather(mi.Vector3f, self.normals, spatial_indices)
        w_l = mi.Frame3f(n).to_local(directions)
        phi = dr.atan2(w_l.y, w_l.x)
        u_idx = dr.minimum(mi.UInt32((phi / (2*dr.pi) + dr.select(phi < 0, 1.0, 0.0)) * self.res_u), self.res_u - 1)
        v_idx = dr.minimum(mi.UInt32(dr.clip(w_l.z, 0.0, 1.0) * self.res_v), self.res_v - 1)
        bin_idx = v_idx * self.res_u + u_idx

        pdf = dr.zeros(mi.Float, dr.width(spatial_indices))
        for i in range(self.n_bins_per_point):
            pdf = dr.select(bin_idx == i, weights[i], pdf)
            
        return dr.select(w_l.z > 0, pdf * (self.n_bins_per_point / (2 * dr.pi)), 0.0)

    def compute_radiance_estimate(self, spatial_indices):
        all_q, res, d_omega = self.get_q_data(spatial_indices), mi.Color3f(0.0), (2 * dr.pi) / self.n_bins_per_point
        for i, q in enumerate(all_q): res += q * self.bin_cosines[i] * d_omega
        return res / dr.pi

    def get_total_visits(self, spatial_indices):
        total = dr.zeros(mi.Float, dr.width(spatial_indices))
        for i in range(self.n_bins_per_point): total += dr.gather(mi.Float, self.visit_counts, spatial_indices * self.n_bins_per_point + i)
        return total

    def get_stats(self):
        v = self.visit_counts
        return {"total_visits": dr.sum(v)[0], "max_q": dr.max(self.sum_r/dr.maximum(v,1.0))[0], "mean_q": dr.sum(self.sum_r)[0]/dr.maximum(dr.sum(v)[0],1.0)}

class DistributeSurfacePointsonScene:
    def __init__(self, scene, n_points, resolution_u=8, resolution_v=8, grid_res=16):
        shapes = [s for s in scene.shapes() if s.emitter() is None]
        n_per = max(1, n_points // len(shapes))
        px, py, pz, nx, ny, nz = [], [], [], [], [], []
        for i, s in enumerate(shapes):
            pcg = mi.PCG32(size=n_per, initstate=i)
            ps = s.sample_position(0.0, mi.Point2f(pcg.next_float32(), pcg.next_float32()))
            px.append(ps.p.x); py.append(ps.p.y); pz.append(ps.p.z); nx.append(ps.n.x); ny.append(ps.n.y); nz.append(ps.n.z)
        self.positions = mi.Point3f(dr.concat(px), dr.concat(py), dr.concat(pz))
        self.normals = mi.Vector3f(dr.concat(nx), dr.concat(ny), dr.concat(nz))
        self.irradiance_volume = SurfaceIrradianceVolume(scene, self.positions, self.normals, resolution_u, resolution_v, grid_res)

    def save(self, path):
        """Saves the sampled positions and normals to a PLY file for visualization."""
        pos, norm = np.array(self.positions), np.array(self.normals)
        with open(path, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {dr.width(self.positions)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write("end_header\n")
            for i in range(dr.width(self.positions)):
                f.write(f"{pos[0, i]} {pos[1, i]} {pos[2, i]} {norm[0, i]} {norm[1, i]} {norm[2, i]}\n")

    def save_hemi(self, path):
        """Saves the hemisphere visualization of the learned Q-values for each point."""
        radius = 10.0
        res_u, res_v = self.irradiance_volume.res_u, self.irradiance_volume.res_v
        n_bins = self.irradiance_volume.n_bins_per_point
        n_points = self.irradiance_volume.n_points

        with open(path, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n_points * n_bins * 4}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float r\nproperty float g\nproperty float b\n")
            f.write(f"element face {n_points * n_bins}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            for i in range(n_points):
                p = dr.gather(mi.Point3f, self.positions, i)
                n = dr.gather(mi.Vector3f, self.normals, i)
                frame = mi.Frame3f(n)
                for j in range(n_bins):
                    u_idx, v_idx = j % res_u, j // res_u
                    for du, dv in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                        phi = (u_idx + du) * (2 * dr.pi / res_u)
                        cos_theta = dr.clip((v_idx + dv) / res_v, 0.0, 1.0)
                        sin_theta = dr.safe_sqrt(1.0 - cos_theta * cos_theta)
                        local_dir = mi.Vector3f(sin_theta * dr.cos(phi), sin_theta * dr.sin(phi), cos_theta)
                        v_pos = p + frame.to_world(local_dir) * radius
                        r, g, b = np.random.rand(3)
                        f.write(f"{v_pos.x[0]} {v_pos.y[0]} {v_pos.z[0]} {r} {g} {b}\n")

            for i in range(n_points * n_bins):
                f.write(f"4 {i*4} {i*4+1} {i*4+2} {i*4+3}\n")

class RLIntegrator(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.n_probes, self.enable_guiding, self.update_q = props.get('n_probes', 1000), props.get('enable_guiding', True), props.get('update_q', True)
        self.resolution_u, self.resolution_v = props.get('resolution_u', 8), props.get('resolution_v', 8)
        self.grid_res = props.get('grid_res', 16)
        self.volume = None

    def sample(self, scene, sampler, ray, medium, active, update_q=True):
        if self.enable_guiding and self.volume is None: self.volume = DistributeSurfacePointsonScene(scene, self.n_probes, self.resolution_u, self.resolution_v, self.grid_res).irradiance_volume
        throughput, result = mi.Spectrum(1.0), mi.Spectrum(0.0)
        prev_idx, prev_dir, has_prev, depth = dr.zeros(mi.UInt32, dr.width(active)), mi.Vector3f(0.0), dr.full(mi.Bool, False, dr.width(active)), 0
        while dr.any(active) and depth < 8:
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            if self.update_q and self.enable_guiding and dr.any(has_prev):
                curr_idx, emitter = self.volume.nearest_point(si.p, si.n), si.emitter(scene)
                L_dir = dr.select(si.is_valid() & (emitter != None), emitter.eval(si, si.is_valid()), 0.0)
                L_ind = dr.select(si.is_valid() & (emitter == None), si.bsdf().eval_diffuse_reflectance(si) * self.volume.compute_radiance_estimate(curr_idx), 0.0)
                self.volume.update(prev_idx, prev_dir, L_dir + L_ind, has_prev)
            
            # Next Event Estimation (NEE)
            if dr.any(active):
                bsdf = si.bsdf(ray)
                emitter_sample, emitter_weight = scene.sample_emitter_direction(si, sampler.next_2d(active), True, active)
                active_nee = active & (mi.luminance(emitter_weight) > 0)
                if dr.any(active_nee):
                    shadow_ray = si.spawn_ray_to(emitter_sample.p)
                    occluded = scene.ray_test(shadow_ray, active_nee)
                    result += dr.select(active_nee & ~occluded, throughput * emitter_weight * bsdf.eval(mi.BSDFContext(), si, si.to_local(emitter_sample.d), active_nee), 0.0)

            if dr.any(active & (si.emitter(scene) != None)): result += dr.select(depth == 0, throughput * si.emitter(scene).eval(si, active), 0.0)
            if not dr.any(active): break
            bsdf, ctx = si.bsdf(ray), mi.BSDFContext()
            if self.enable_guiding:
                curr_idx = self.volume.nearest_point(si.p, si.n)
                alpha = dr.select((dr.dot(si.n, dr.gather(mi.Vector3f, self.volume.normals, curr_idx)) > 0.0) & (self.volume.get_q_sum(curr_idx) > 1e-4) & (self.volume.get_total_visits(curr_idx) > 100), 0.4, 0.0)
                wo_rl, _ = self.volume.sample_direction(curr_idx, sampler.next_2d(active))
                bs_s, bs_w = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)
                direction = dr.select(sampler.next_1d(active) < alpha, wo_rl, si.to_world(bs_s.wo))
                pdf_mix = alpha * self.volume.pdf_direction(curr_idx, direction) + (1.0 - alpha) * bsdf.pdf(ctx, si, si.to_local(direction), active)
                throughput *= dr.select(alpha > 0, bsdf.eval(ctx, si, si.to_local(direction), active) * dr.maximum(0.0, dr.dot(direction, si.n)) / dr.maximum(pdf_mix, 1e-7), bs_w)
                prev_idx, prev_dir, has_prev = curr_idx, direction, active
            else:
                bs_s, bs_w = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)
                direction, throughput, has_prev = si.to_world(bs_s.wo), throughput * bs_w, dr.full(mi.Bool, False, dr.width(active))
            ray, active, depth = si.spawn_ray(direction), active & dr.any(throughput != 0.0), depth + 1
        return result, active, []

mi.register_integrator("rl_integrator", lambda props: RLIntegrator(props))
