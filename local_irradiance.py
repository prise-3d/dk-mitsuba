import drjit as dr
import mitsuba as mi
import numpy as np


class SurfaceIrradianceVolume:
    """
    Structure de données pour le 'Learning Light Transport'.
    
    Elle gère un ensemble de points discrets sur les surfaces (positions + normales).
    À chaque point est attachée une hémisphère discrétisée (tableau 2D) stockant :
    - La somme des récompenses (pour calculer Q)
    - Le nombre de visites
    """

    def __init__(self, positions, normals, resolution_u=8, resolution_v=8):
        """
        Initialise la structure.

        :param positions: [N, 3] Array des positions des points sur la surface.
        :param normals:   [N, 3] Array des normales associées.
        :param resolution_u: Résolution azimutale (phi) de l'hémisphère.
        :param resolution_v: Résolution zénithale (theta) de l'hémisphère.
        """
        self.positions = mi.Point3f(positions)
        self.normals = mi.Vector3f(normals)
        
        # Dimensions
        self.n_points = dr.width(self.positions)
        self.res_u = resolution_u
        self.res_v = resolution_v
        self.n_bins_per_point = resolution_u * resolution_v
        
        # Stockage plat pour Dr.Jit (Structure of Arrays)
        # Taille totale = n_points * n_bins_per_point
        total_size = self.n_points * self.n_bins_per_point
        
        # On stocke la somme des valeurs pour permettre des mises à jour atomiques faciles
        # Q = sum_values / visit_counts
        self.sum_values = dr.zeros(mi.Float, total_size)
        self.visit_counts = dr.zeros(mi.Float, total_size) # Float pour faciliter les divisions

    def get_q_value(self, spatial_indices, directions):
        """
        Récupère la valeur Q (moyenne) pour des points spatiaux donnés et des directions données.
        """
        bin_indices = self._get_bin_indices(spatial_indices, directions)
        flat_indices = spatial_indices * self.n_bins_per_point + bin_indices
        total_value = dr.gather(mi.Float, self.sum_values, flat_indices)
        count = dr.gather(mi.Float, self.visit_counts, flat_indices)
        return dr.select(count > 0, total_value / count, 0.0)

    def update(self, spatial_indices, directions, rewards):
        """
        Met à jour les statistiques (accumule la récompense et incrémente le compteur).
        """
        bin_indices = self._get_bin_indices(spatial_indices, directions)
        flat_indices = spatial_indices * self.n_bins_per_point + bin_indices
        dr.scatter_reduce(dr.ReduceOp.Add, self.sum_values, rewards, flat_indices)
        dr.scatter_reduce(dr.ReduceOp.Add, self.visit_counts, 1.0, flat_indices)

    def _get_bin_indices(self, spatial_indices, directions):
        """Transforme Direction Monde -> Index de Bin Local (u, v)"""
        n = dr.gather(mi.Vector3f, self.normals, spatial_indices)
        frame = mi.Frame3f(n)
        w_local = frame.to_local(directions)
        cos_theta = dr.maximum(0.0, dr.minimum(1.0, w_local.z))
        phi = dr.atan2(w_local.y, w_local.x)
        phi = dr.select(phi < 0, phi + 2 * dr.pi, phi)
        u_cont = (phi / (2 * dr.pi)) * self.res_u
        u_idx = dr.minimum(mi.UInt32(u_cont), self.res_u - 1)
        v_cont = cos_theta * self.res_v
        v_idx = dr.minimum(mi.UInt32(v_cont), self.res_v - 1)
        return v_idx * self.res_u + u_idx
    
    def nearest_point(self, positions):
        """Trouve les indices des points les plus proches (Vectorisé)."""
        n_queries = dr.width(positions)
        min_dist2 = dr.full(mi.Float, 1e30, n_queries)
        best_indices = dr.zeros(mi.UInt32, n_queries)
        for i in range(self.n_points):
            p_i = dr.gather(mi.Point3f, self.positions, i)
            dist2 = dr.squared_norm(positions - p_i)
            closer = dist2 < min_dist2
            min_dist2 = dr.select(closer, dist2, min_dist2)
            best_indices = dr.select(closer, mi.UInt32(i), best_indices)
        return best_indices

    def get_q_data(self, spatial_indices):
        """Récupère les valeurs Q pour tous les bins."""
        all_q = []
        for i in range(self.n_bins_per_point):
            flat_indices = spatial_indices * self.n_bins_per_point + i
            val = dr.gather(mi.Float, self.sum_values, flat_indices)
            count = dr.gather(mi.Float, self.visit_counts, flat_indices)
            all_q.append(dr.select(count > 0, val / count, 0.0))
        return all_q

    def sample_direction(self, spatial_indices, sample):
        """Échantillonne une direction proportionnelle aux valeurs Q."""
        n_queries = dr.width(spatial_indices)
        all_q = self.get_q_data(spatial_indices)
        q_sum = dr.zeros(mi.Float, n_queries)
        for q in all_q: q_sum += q
        epsilon = 0.1
        weights = [dr.select(q_sum > 0, (1.0 - epsilon) * q / q_sum + epsilon / self.n_bins_per_point, 1.0 / self.n_bins_per_point) for q in all_q]
        cum_weights = []
        curr = dr.zeros(mi.Float, n_queries)
        for w in weights:
            curr += w
            cum_weights.append(curr)
        u = sample.x
        bin_idx = dr.zeros(mi.UInt32, n_queries)
        for i in range(self.n_bins_per_point - 1):
            bin_idx = dr.select(u > cum_weights[i], mi.UInt32(i + 1), bin_idx)
        v_idx, u_idx = bin_idx // self.res_u, bin_idx % self.res_u
        
        prev_c = dr.zeros(mi.Float, n_queries)
        next_c = dr.zeros(mi.Float, n_queries)
        for i in range(self.n_bins_per_point):
            prev_c = dr.select(bin_idx == i, (cum_weights[i-1] if i > 0 else 0.0), prev_c)
            next_c = dr.select(bin_idx == i, cum_weights[i], next_c)

        u_local = (u - prev_c) / dr.maximum(next_c - prev_c, 1e-7)
        phi = (mi.Float(u_idx) + u_local) * (2 * dr.pi / self.res_u)
        cos_theta = (mi.Float(v_idx) + sample.y) * (1.0 / self.res_v)
        sin_theta = dr.safe_sqrt(1.0 - cos_theta * cos_theta)
        w_local = mi.Vector3f(sin_theta * dr.cos(phi), sin_theta * dr.sin(phi), cos_theta)
        n = dr.gather(mi.Vector3f, self.normals, spatial_indices)
        direction = mi.Frame3f(n).to_world(w_local)
        
        pdf = dr.zeros(mi.Float, n_queries)
        for i in range(self.n_bins_per_point):
            pdf = dr.select(bin_idx == i, weights[i], pdf)
        pdf *= (self.n_bins_per_point / (2 * dr.pi))
        return direction, pdf

    def compute_radiance_estimate(self, spatial_indices):
        """Estime la radiance sortante diffuse via l'irradiance apprise."""
        n_queries = dr.width(spatial_indices)
        all_q = self.get_q_data(spatial_indices)
        d_phi, d_cos = 2 * dr.pi / self.res_u, 1.0 / self.res_v
        irradiance = dr.zeros(mi.Float, n_queries)
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

    def save(self, filename, radius=0.01, scale=1.0):
        p_np = np.array(self.positions).T * scale
        n_np = np.array(self.normals).T
        n_probes = p_np.shape[0]
        with open(filename, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {2 * n_probes}\nproperty float x\nproperty float y\nproperty float z\nelement edge {n_probes}\nproperty int vertex1\nproperty int vertex2\nend_header\n")
            for i in range(n_probes):
                f.write(f"{p_np[i,0]} {p_np[i,1]} {p_np[i,2]}\n")
                f.write(f"{p_np[i,0]+n_np[i,0]*radius*scale} {p_np[i,1]+n_np[i,1]*radius*scale} {p_np[i,2]+n_np[i,2]*radius*scale}\n")
            for i in range(n_probes): f.write(f"{2*i} {2*i+1}\n")


class RLIntegrator(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.n_probes = props.get('n_probes', 1000)
        self.enable_guiding = props.get('enable_guiding', True)
        self.volume = None

    def sample(self, scene, sampler, ray, medium, active):
        if self.enable_guiding and self.volume is None:
             self.volume = DistributeSurfacePointsonScene(scene, self.n_probes).irradiance_volume

        throughput, result = mi.Spectrum(1.0), mi.Spectrum(0.0)
        prev_idx, prev_dir = dr.zeros(mi.UInt32, dr.width(active)), mi.Vector3f(0.0)
        has_prev = dr.full(mi.Bool, False, dr.width(active))
        depth = 0
        
        while dr.any(active) and depth < 5:
            # Intersection with the scene
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()


            # Mise à jour de la valeur Q pour l'étape précédente
            if self.enable_guiding and dr.any(has_prev):
                curr_idx = self.volume.nearest_point(si.p)
                emitter = si.emitter(scene)
                has_emitter = active & (emitter != None)
                reward = dr.select(has_emitter, dr.mean(emitter.eval(si, has_emitter)), 0.0)
                reward += self.volume.compute_radiance_estimate(curr_idx)
                self.volume.update(prev_idx, prev_dir, reward)

            # Contribution directe des émetteurs    
            emitter = si.emitter(scene)
            has_emitter = active & (emitter != None)
            if dr.any(has_emitter):
                result += throughput * emitter.eval(si, has_emitter)
            
            # Terminaison si aucune contribution n'est possible
            if not dr.any(active): break

            
            if self.enable_guiding:
                # Echantillonnage guidé par RL
                curr_idx = self.volume.nearest_point(si.p)
                use_q = sampler.next_1d(active) < 0.8
                wo_q, pdf_q = self.volume.sample_direction(curr_idx, sampler.next_2d(active))
            else:
                # Pas de guidage, échantillonnage BSDF classique
                curr_idx = dr.zeros(mi.UInt32, dr.width(active))
                use_q = dr.full(mi.Bool, False, dr.width(active))
                wo_q = mi.Vector3f(0.0)
                pdf_q = dr.zeros(mi.Float, dr.width(active))
            bsdf = si.bsdf(ray)
            ctx = mi.BSDFContext()
            bs_sample, bs_weight = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)
            
            # Choix entre direction guidée par RL ou échantillonnage BSDF
            direction = dr.select(use_q, wo_q, bs_sample.wo)
            # Calcul du throughput en fonction du choix d'échantillonnage
            pdf = dr.select(use_q, pdf_q, bs_sample.pdf)
            # Mise à jour du throughput
            throughput *= bsdf.eval(ctx, si, direction, active) * dr.abs(dr.dot(direction, si.n)) / dr.maximum(pdf, 1e-7)
            ray = si.spawn_ray(direction)
            if self.enable_guiding:
                prev_idx, prev_dir, has_prev = curr_idx, direction, active
            else:
                has_prev = dr.full(mi.Bool, False, dr.width(active))
            active &= dr.any(throughput != 0.0)
            depth += 1
            
        return result, active, []

mi.register_integrator("rl_integrator", lambda props: RLIntegrator(props))

