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

    def nearest_point(self, p):
        p_rel = (p - self.grid_min) / self.grid_size
        ix = dr.clip(mi.UInt32(p_rel.x * self.grid_res), 0, self.grid_res - 1)
        iy = dr.clip(mi.UInt32(p_rel.y * self.grid_res), 0, self.grid_res - 1)
        iz = dr.clip(mi.UInt32(p_rel.z * self.grid_res), 0, self.grid_res - 1)
        idx = ix + iy * self.grid_res + iz * (self.grid_res**2)
        return dr.gather(mi.UInt32, self.grid_data, idx)

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

    def sample_direction(self, spatial_indices, sample):
        all_q, epsilon = self.get_q_data(spatial_indices), 0.1
        q_cos = [mi.luminance(q) * self.bin_cosines[i] for i, q in enumerate(all_q)]
        q_sum = dr.zeros(mi.Float, dr.width(spatial_indices))
        for val in q_cos: q_sum += val
        weights = [dr.select(q_sum > 1e-6, (1.0 - epsilon) * q_c / q_sum + epsilon / self.n_bins_per_point, 1.0 / self.n_bins_per_point) for q_c in q_cos]
        cum_w, curr = [], dr.zeros(mi.Float, dr.width(spatial_indices))
        for w in weights: curr += w; cum_w.append(curr)
        bin_idx = dr.zeros(mi.UInt32, dr.width(spatial_indices))
        for i in range(self.n_bins_per_point - 1): bin_idx = dr.select(sample.x > cum_w[i], mi.UInt32(i + 1), bin_idx)
        prev_c = dr.zeros(mi.Float, dr.width(spatial_indices))
        for i in range(self.n_bins_per_point): prev_c = dr.select(bin_idx == i, (cum_w[i-1] if i > 0 else 0.0), prev_c)
        u_l = dr.clip((sample.x - prev_c) / dr.maximum(dr.gather(mi.Float, dr.concat(cum_w), bin_idx) - prev_c, 1e-7), 0.0, 1.0)
        phi, cos_theta = (mi.Float(bin_idx % self.res_u) + u_l) * (2 * dr.pi / self.res_u), dr.clip((mi.Float(bin_idx // self.res_u) + sample.y) / self.res_v, 0.0, 1.0)
        sin_theta = dr.safe_sqrt(1.0 - cos_theta * cos_theta)
        direction = mi.Frame3f(dr.gather(mi.Vector3f, self.normals, spatial_indices)).to_world(mi.Vector3f(sin_theta * dr.cos(phi), sin_theta * dr.sin(phi), cos_theta))
        pdf = dr.zeros(mi.Float, dr.width(spatial_indices))
        for i in range(self.n_bins_per_point): pdf = dr.select(bin_idx == i, weights[i], pdf)
        return direction, pdf * (self.n_bins_per_point / (2 * dr.pi))

    def pdf_direction(self, spatial_indices, directions):
        n = dr.gather(mi.Vector3f, self.normals, spatial_indices)
        w_l = mi.Frame3f(n).to_local(directions)
        phi = dr.atan2(w_l.y, w_l.x)
        bin_idx = dr.minimum(mi.UInt32(dr.clip(w_l.z, 0.0, 1.0) * self.res_v), self.res_v - 1) * self.res_u + dr.minimum(mi.UInt32((phi / (2*dr.pi) + dr.select(phi < 0, 1.0, 0.0)) * self.res_u), self.res_u - 1)
        all_q, epsilon = self.get_q_data(spatial_indices), 0.1
        q_cos = [mi.luminance(q) * self.bin_cosines[i] for i, q in enumerate(all_q)]
        q_sum = dr.zeros(mi.Float, dr.width(spatial_indices))
        for val in q_cos: q_sum += val
        pdf = dr.zeros(mi.Float, dr.width(spatial_indices))
        for i in range(self.n_bins_per_point):
            w = dr.select(q_sum > 1e-6, (1.0 - epsilon) * q_cos[i] / q_sum + epsilon / self.n_bins_per_point, 1.0 / self.n_bins_per_point)
            pdf = dr.select(bin_idx == i, w, pdf)
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
    def __init__(self, scene, n_points):
        shapes = [s for s in scene.shapes() if s.emitter() is None]
        n_per = max(1, n_points // len(shapes))
        px, py, pz, nx, ny, nz = [], [], [], [], [], []
        for i, s in enumerate(shapes):
            pcg = mi.PCG32(size=n_per, initstate=i)
            ps = s.sample_position(0.0, mi.Point2f(pcg.next_float32(), pcg.next_float32()))
            px.append(ps.p.x); py.append(ps.p.y); pz.append(ps.p.z); nx.append(ps.n.x); ny.append(ps.n.y); nz.append(ps.n.z)
        self.irradiance_volume = SurfaceIrradianceVolume(scene, mi.Point3f(dr.concat(px), dr.concat(py), dr.concat(pz)), mi.Vector3f(dr.concat(nx), dr.concat(ny), dr.concat(nz)))

class RLIntegrator(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.n_probes, self.enable_guiding, self.update_q = props.get('n_probes', 1000), props.get('enable_guiding', True), props.get('update_q', True)
        self.volume = None

    def sample(self, scene, sampler, ray, medium, active, update_q=True):
        if self.enable_guiding and self.volume is None: self.volume = DistributeSurfacePointsonScene(scene, self.n_probes).irradiance_volume
        throughput, result = mi.Spectrum(1.0), mi.Spectrum(0.0)
        prev_idx, prev_dir, has_prev, depth = dr.zeros(mi.UInt32, dr.width(active)), mi.Vector3f(0.0), dr.full(mi.Bool, False, dr.width(active)), 0
        while dr.any(active) and depth < 8:
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            if self.update_q and self.enable_guiding and dr.any(has_prev):
                curr_idx, emitter = self.volume.nearest_point(si.p), si.emitter(scene)
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

            if dr.any(active & (si.emitter(scene) != None)): result += throughput * si.emitter(scene).eval(si, active)
            if not dr.any(active): break
            bsdf, ctx = si.bsdf(ray), mi.BSDFContext()
            if self.enable_guiding:
                curr_idx = self.volume.nearest_point(si.p)
                alpha = dr.select((dr.dot(si.n, dr.gather(mi.Vector3f, self.volume.normals, curr_idx)) > 0.5) & (self.volume.get_q_sum(curr_idx) > 1e-4) & (self.volume.get_total_visits(curr_idx) > 100), 0.4, 0.0)
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
