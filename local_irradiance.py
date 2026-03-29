import drjit as dr
import mitsuba as mi
import numpy as np

# Initialiser le variant par défaut si nécessaire
if not mi.variant():
    mi.set_variant('llvm_ad_rgb')

class SurfaceIrradianceVolume:
    def __init__(self, positions, normals, resolution_u=8, resolution_v=8):
        self.positions = mi.Point3f(positions)
        self.normals = mi.Vector3f(normals)
        self.n_points = dr.width(self.positions)
        self.res_u, self.res_v = resolution_u, resolution_v
        self.n_bins_per_point = resolution_u * resolution_v
        total_size = self.n_points * self.n_bins_per_point
        self.sum_values = dr.zeros(mi.Float, total_size)
        self.visit_counts = dr.zeros(mi.Float, total_size)

    def _get_bin_indices(self, spatial_indices, directions):
        n = dr.gather(mi.Vector3f, self.normals, spatial_indices)
        w_local = mi.Frame3f(n).to_local(directions)
        cos_theta = dr.maximum(0.0, dr.minimum(1.0, w_local.z))
        phi = dr.atan2(w_local.y, w_local.x)
        phi = dr.select(phi < 0, phi + 2 * dr.pi, phi)
        u_idx = dr.minimum(mi.UInt32((phi / (2 * dr.pi)) * self.res_u), self.res_u - 1)
        v_idx = dr.minimum(mi.UInt32(cos_theta * self.res_v), self.res_v - 1)
        return v_idx * self.res_u + u_idx

    def update(self, spatial_indices, directions, rewards, mask=True):
        bin_idx = self._get_bin_indices(spatial_indices, directions)
        flat_idx = spatial_indices * self.n_bins_per_point + bin_idx
        dr.scatter_reduce(dr.ReduceOp.Add, self.sum_values, rewards, flat_idx, mask)
        dr.scatter_reduce(dr.ReduceOp.Add, self.visit_counts, 1.0, flat_idx, mask)

    def nearest_point(self, positions):
        n_queries = dr.width(positions)
        min_dist2 = dr.full(mi.Float, 1e30, n_queries)
        best_idx = dr.zeros(mi.UInt32, n_queries)
        for i in range(self.n_points):
            p_i = dr.gather(mi.Point3f, self.positions, i)
            dist2 = dr.squared_norm(positions - p_i)
            closer = dist2 < min_dist2
            min_dist2 = dr.select(closer, dist2, min_dist2)
            best_idx = dr.select(closer, mi.UInt32(i), best_idx)
        return best_idx

    def get_q_data(self, spatial_indices):
        all_q = []
        for i in range(self.n_bins_per_point):
            flat_idx = spatial_indices * self.n_bins_per_point + i
            val = dr.gather(mi.Float, self.sum_values, flat_idx)
            count = dr.gather(mi.Float, self.visit_counts, flat_idx)
            all_q.append(dr.select(count > 0, val / count, 0.0))
        return all_q

    def sample_direction(self, spatial_indices, sample):
        n_queries = dr.width(spatial_indices)
        all_q = self.get_q_data(spatial_indices)
        q_sum = dr.zeros(mi.Float, n_queries)
        for q in all_q: q_sum += q
        epsilon = 0.1
        
        # Probabilités par bin
        weights = [dr.select(q_sum > 0, (1.0 - epsilon) * q / q_sum + epsilon / self.n_bins_per_point, 1.0 / self.n_bins_per_point) for q in all_q]
        
        cum_weights = []
        curr = dr.zeros(mi.Float, n_queries)
        for w in weights:
            curr += w
            cum_weights.append(dr.detach(curr))
        
        u = sample.x
        bin_idx = dr.zeros(mi.UInt32, n_queries)
        for i in range(self.n_bins_per_point - 1):
            bin_idx = dr.select(u > cum_weights[i], mi.UInt32(i + 1), bin_idx)
        
        # Sélection robuste des cumulative weights et probabilités sans concat
        prev_c = dr.zeros(mi.Float, n_queries)
        next_c = dr.zeros(mi.Float, n_queries)
        prob_bin = dr.zeros(mi.Float, n_queries)
        
        for i in range(self.n_bins_per_point):
            mask = (bin_idx == mi.UInt32(i))
            if i > 0:
                prev_c = dr.select(mask, cum_weights[i-1], prev_c)
            next_c = dr.select(mask, cum_weights[i], next_c)
            prob_bin = dr.select(mask, weights[i], prob_bin)
            
        u_local = (u - prev_c) / dr.maximum(next_c - prev_c, 1e-7)
        
        v_idx, u_idx = bin_idx // self.res_u, bin_idx % self.res_u
        phi = (mi.Float(u_idx) + u_local) * (2 * dr.pi / self.res_u)
        cos_theta = (mi.Float(v_idx) + sample.y) * (1.0 / self.res_v)
        sin_theta = dr.safe_sqrt(1.0 - cos_theta * cos_theta)
        w_local = mi.Vector3f(sin_theta * dr.cos(phi), sin_theta * dr.sin(phi), cos_theta)
        
        n = dr.gather(mi.Vector3f, self.normals, spatial_indices)
        direction = mi.Frame3f(n).to_world(w_local)
        
        pdf = prob_bin * (self.n_bins_per_point / (2 * dr.pi))
        
        return direction, pdf

    def compute_radiance_estimate(self, spatial_indices):
        all_q = self.get_q_data(spatial_indices)
        d_phi, d_cos = 2 * dr.pi / self.res_u, 1.0 / self.res_v
        irradiance = dr.zeros(mi.Float, dr.width(spatial_indices))
        for i in range(self.n_bins_per_point):
            cos_j = (mi.Float(i // self.res_u) + 0.5) / self.res_v
            irradiance += all_q[i] * cos_j * d_phi * d_cos
        return irradiance / dr.pi

class DistributeSurfacePointsonScene:
    def __init__(self, scene, n_points):
        self.scene = scene
        self.positions, self.normals = self._sample_points_on_scene(n_points)
        self.irradiance_volume = SurfaceIrradianceVolume(self.positions, self.normals)

    def _sample_points_on_scene(self, n_points):
        shapes = self.scene.shapes()
        n_per_shape = max(1, n_points // len(shapes))
        px, py, pz, nx, ny, nz = [], [], [], [], [], []
        for i, shape in enumerate(shapes):
            pcg = mi.PCG32(size=n_per_shape, initstate=i)
            ps = shape.sample_position(0.0, mi.Point2f(pcg.next_float32(), pcg.next_float32()))
            px.append(ps.p.x); py.append(ps.p.y); pz.append(ps.p.z)
            nx.append(ps.n.x); ny.append(ps.n.y); nz.append(ps.n.z)
        return mi.Point3f(dr.concat(px), dr.concat(py), dr.concat(pz)), mi.Vector3f(dr.concat(nx), dr.concat(ny), dr.concat(nz))

class RLIntegrator(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.n_probes = props.get('n_probes', 1000)
        self.volume = None

    def sample(self, scene, sampler, ray, medium, active):
        if self.volume is None:
             self.volume = DistributeSurfacePointsonScene(scene, self.n_probes).irradiance_volume

        throughput, result = mi.Spectrum(1.0), mi.Spectrum(0.0)
        prev_idx, prev_dir = dr.zeros(mi.UInt32, dr.width(active)), mi.Vector3f(0.0)
        has_prev = mi.Mask(False)
        depth = 0
        
        while dr.any(active) and depth < 5:
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            
            update_mask = active & has_prev
            if dr.any(update_mask):
                curr_idx = self.volume.nearest_point(si.p)
                emitter = si.emitter(scene)
                reward = dr.select(emitter != None, dr.mean(emitter.eval(si, update_mask)), 0.0)
                reward += self.volume.compute_radiance_estimate(curr_idx)
                self.volume.update(prev_idx, prev_dir, mi.Float(reward), update_mask)

            emitter = si.emitter(scene)
            active_emitter = active & (emitter != None)
            if dr.any(active_emitter):
                result += dr.select(active_emitter, throughput * emitter.eval(si, active_emitter), 0.0)
            
            if not dr.any(active): break
                
            curr_idx = self.volume.nearest_point(si.p)
            use_q = sampler.next_1d(active) < 0.8
            wo_q, pdf_q = self.volume.sample_direction(curr_idx, sampler.next_2d(active))
            
            bsdf = si.bsdf(ray)
            ctx = mi.BSDFContext()
            bs_sample, bs_weight = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)
            
            direction = dr.select(use_q, wo_q, bs_sample.wo)
            pdf = dr.select(use_q, pdf_q, bs_sample.pdf)
            pdf = dr.maximum(pdf, 1e-7)
            
            throughput *= bsdf.eval(ctx, si, direction, active) * dr.abs(dr.dot(direction, si.n)) / pdf
            ray = si.spawn_ray(direction)
            
            prev_idx, prev_dir, has_prev = curr_idx, direction, mi.Mask(active)
            active &= (dr.max(throughput) > 0.0)
            depth += 1
            
        return result, active, []

mi.register_integrator("rl_integrator", lambda props: RLIntegrator(props))
