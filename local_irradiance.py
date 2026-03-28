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
        
        :param spatial_indices: [k] Indices des points (int) dans le tableau self.positions.
        :param directions:      [k, 3] Directions (monde) normalisées.
        :return:                [k] Valeurs Q estimées.
        """
        # 1. Calculer l'index plat dans le tableau de données
        bin_indices = self._get_bin_indices(spatial_indices, directions)
        flat_indices = spatial_indices * self.n_bins_per_point + bin_indices
        
        # 2. Gather les données
        total_value = dr.gather(mi.Float, self.sum_values, flat_indices)
        count = dr.gather(mi.Float, self.visit_counts, flat_indices)
        
        # 3. Calculer Q (moyenne). Si count == 0, retourne 0.
        return dr.select(count > 0, total_value / count, 0.0)

    def update(self, spatial_indices, directions, rewards):
        """
        Met à jour les statistiques (accumule la récompense et incrémente le compteur).
        Thread-safe via scatter_reduce (atomic add).

        :param spatial_indices: [k] Indices des points concernés.
        :param directions:      [k, 3] Directions incidentes (monde).
        :param rewards:         [k] Valeur reçue (radiance/throughput).
        """
        # 1. Identifier les bins concernés
        bin_indices = self._get_bin_indices(spatial_indices, directions)
        flat_indices = spatial_indices * self.n_bins_per_point + bin_indices
        
        # 2. Accumulation atomique
        dr.scatter_reduce(dr.ReduceOp.Add, self.sum_values, rewards, flat_indices)
        dr.scatter_reduce(dr.ReduceOp.Add, self.visit_counts, 1.0, flat_indices)

    def _get_bin_indices(self, spatial_indices, directions):
        """
        Méthode interne : Transforme Direction Monde -> Index de Bin Local (u, v)
        en utilisant la normale stockée pour chaque point spatial.
        """
        # Récupérer la normale associée à chaque point spatial
        n = dr.gather(mi.Vector3f, self.normals, spatial_indices)
        
        # Créer un repère local (Frame) à partir de la normale
        frame = mi.Frame3f(n)
        
        # Convertir la direction monde en direction locale
        w_local = frame.to_local(directions)
        
        # --- Projection Hémisphérique ---
        # w_local.z est le cos(theta) par rapport à la normale.
        # On suppose que les directions viennent du dessus (w_local.z > 0).
        # Si w_local.z < 0 (sous la surface), on peut clamper ou ignorer.
        
        # Coordonnées sphériques
        # Theta : angle avec la normale [0, pi/2] -> cos_theta [0, 1]
        cos_theta = dr.maximum(0.0, dr.minimum(1.0, w_local.z))
        
        # Phi : angle azimutal [-pi, pi]
        phi = dr.atan2(w_local.y, w_local.x)
        phi = dr.select(phi < 0, phi + 2 * dr.pi, phi) # Map [0, 2pi]
        
        # --- Binning ---
        
        # Mapping u (phi) : [0, 2pi] -> [0, res_u]
        u_cont = (phi / (2 * dr.pi)) * self.res_u
        u_idx = dr.minimum(mi.UInt32(u_cont), self.res_u - 1)
        
        # Mapping v (theta) : cos_theta [0, 1] -> [0, res_v]
        # Note: On map souvent cos_theta directement pour avoir des aires égales
        # si on découpe en anneaux, ou theta linéairement.
        # Ici mapping linéaire de cos_theta pour simplicité (Nusselt-like)
        # v=0 -> horizon, v=max -> zénith (normale)
        v_cont = cos_theta * self.res_v
        v_idx = dr.minimum(mi.UInt32(v_cont), self.res_v - 1)
        
        # Index 1D dans l'hémisphère
        return v_idx * self.res_u + u_idx
    
    def nearest_point(self, position):
        """
        Trouve l'indice du point le plus proche de la position donnée.
        (devra être optimisé avec une structure de données spatiale)
        """
        p = mi.Point3f(position)
        # Calculer les distances au carré pour éviter la racine carrée
        diff = self.positions - p
        dist2 = dr.squared_norm(diff)

        # Trouver l'indice du minimum (argmin manuel en drjit)
        min_val = dr.min(dist2)
        indices = dr.arange(mi.UInt32, dr.width(dist2))
        return dr.min(dr.select(dist2 == min_val, indices, dr.width(dist2)))

# --- Zone de Test Rapide ---
if __name__ == "__main__":
    mi.set_variant('llvm_ad_rgb') # Ou scalar_rgb
    
    # 1. Création de quelques points (ex: 2 points sur un plan Z=0)
    positions = mi.Point3f([0, 10], [0, 0], [0, 0])
    normals = mi.Vector3f([0, 0], [0, 0], [1, 1]) # Normales vers Z+
    
    # Résolution faible pour le test (2x2 bins)
    # Bins:
    # v=0 (rasant), v=1 (zénith)
    # u=0 (0-180°), u=1 (180-360°)
    vol = SurfaceIrradianceVolume(positions, normals, 2, 2)
    
    # 2. Test d'Update
    # On éclaire le point 0 avec une direction verticale (Z+)
    # C'est le bin (v=1, u=quelconque)
    idx = mi.Int32([0])
    dir_up = mi.Vector3f([0, 0, 1])
    reward = mi.Float([10.0])
    
    print("Update point 0, dir Z+ avec valeur 10.0")
    vol.update(idx, dir_up, reward)
    
    # 3. Test de Query (Même direction)
    q = vol.get_q_value(idx, dir_up)
    print(f"Q-value (Point 0, Z+): {q}") # Devrait être 10.0
    
    # 4. Test d'Update cumulatif (Moyenne)
    print("Update point 0, dir Z+ avec valeur 20.0")
    vol.update(idx, dir_up, mi.Float([20.0]))
    
    q_new = vol.get_q_value(idx, dir_up)
    print(f"Q-value (Point 0, Z+): {q_new}") # Devrait être (10+20)/2 = 15.0
    
    # 5. Test Direction différente (Point 0, direction X+)
    # X+ est rasant (cos_theta ~ 0), donc v=0
    dir_side = mi.Vector3f([1, 0, 0]) 
    q_side = vol.get_q_value(idx, dir_side)
    print(f"Q-value (Point 0, X+): {q_side}") # Devrait être 0.0 (bin différent)
    
    print("Tests terminés.")

    # make a list of 10 randoms points with normals up
    
    n_points = 10
    positions = np.random.rand(n_points, 3) * 10.0 # Random positions in a 10x10x10 cube
    positions[:, 2] = 0 # All points on the plane Z=0
    normals = np.tile([0, 0, 1], (n_points, 1)) # Normales vers Z+
    
    positions = np.ascontiguousarray(positions, dtype=np.float32).T
    normals = np.ascontiguousarray(normals, dtype=np.float32).T
    vol = SurfaceIrradianceVolume(positions, normals, 8, 8) # 8x8 bins

    # chose a new random point with normal up
    new_point = np.random.rand(3) * 10.0
    new_point[2] = 0 # On the plane
    new_normal = np.array([0, 0, 1])    

    nearest_idx = vol.nearest_point(new_point)
    # On utilise gather pour récupérer la position correspondant à l'indice trouvé
    nearest_pos = dr.gather(mi.Point3f, vol.positions, nearest_idx)
    
    print(f"New point: {new_point}, Nearest point index: {nearest_idx}, Nearest point position: {nearest_pos}")

    # Vérification avec numpy pour être sûr
    all_pos_np = np.array(vol.positions).T # Convert back to (N, 3)
    dists = np.linalg.norm(all_pos_np - new_point, axis=1)
    np_nearest_idx = np.argmin(dists)
    print(f"Numpy nearest index: {np_nearest_idx}, Numpy nearest position: {all_pos_np[np_nearest_idx]}")
    
    if int(nearest_idx[0]) == np_nearest_idx:
        print("SUCCESS: Nearest point matches numpy result.")
    else:
        print("FAILURE: Nearest point does not match numpy result.")

    # with numpy, we compute a tabular of the distance from the new point
    # to all the points in the volume and print all the distances and the nearest point
    print("Distances from new point to all points in the volume:")
    for i in range(n_points):
        dist = np.linalg.norm(all_pos_np[i] - new_point)
        print(f"Point {i}: Position {all_pos_np[i]}, Distance: {dist}") 
        


