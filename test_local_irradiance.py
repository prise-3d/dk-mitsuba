import mitsuba as mi
# Configure Mitsuba variant for tests
mi.set_variant('llvm_ad_rgb')

import drjit as dr
import numpy as np
import pytest
from local_irradiance import SurfaceIrradianceVolume


def test_initialization():
    """Vérifie que la structure est bien initialisée avec les bonnes dimensions."""
    n_points = 5
    positions = np.zeros((3, n_points), dtype=np.float32)
    normals = np.zeros((3, n_points), dtype=np.float32)
    normals[2, :] = 1.0 # Toutes les normales vers Z+
    
    res_u, res_v = 4, 8
    vol = SurfaceIrradianceVolume(positions, normals, res_u, res_v)
    
    assert vol.n_points == n_points
    assert vol.res_u == res_u
    assert vol.res_v == res_v
    assert vol.n_bins_per_point == res_u * res_v
    assert dr.width(vol.sum_values) == n_points * res_u * res_v
    assert dr.all(vol.visit_counts == 0)

def test_update_and_query_averaging():
    """Vérifie que les mises à jour accumulent correctement et calculent la moyenne."""
    positions = mi.Point3f([0], [0], [0])
    normals = mi.Vector3f([0], [0], [1])
    vol = SurfaceIrradianceVolume(positions, normals, 2, 2)
    
    idx = mi.UInt32([0])
    direction = mi.Vector3f([0], [0], [1])
    
    # Premier update
    vol.update(idx, direction, mi.Float([10.0]))
    q = vol.get_q_value(idx, direction)
    assert dr.allclose(q, 10.0)
    
    # Deuxième update (même bin)
    vol.update(idx, direction, mi.Float([20.0]))
    q = vol.get_q_value(idx, direction)
    assert dr.allclose(q, 15.0) # (10 + 20) / 2

def test_directional_bins():
    """Vérifie que des directions différentes tombent dans des bins différents."""
    positions = mi.Point3f([0], [0], [0])
    normals = mi.Vector3f([0], [0], [1])
    vol = SurfaceIrradianceVolume(positions, normals, 4, 4)
    
    idx = mi.UInt32([0])
    dir_up = mi.Vector3f([0, 0, 1])
    dir_side = mi.Vector3f([1, 0, 0])
    
    vol.update(idx, dir_up, mi.Float([100.0]))
    
    q_up = vol.get_q_value(idx, dir_up)
    q_side = vol.get_q_value(idx, dir_side)
    
    assert dr.allclose(q_up, 100.0)
    assert dr.allclose(q_side, 0.0)

def test_nearest_point():
    """Vérifie que nearest_point trouve bien le point le plus proche (comparaison avec numpy)."""
    n_points = 20
    # On génère des points aléatoires
    pos_np = np.random.rand(n_points, 3).astype(np.float32) * 10.0
    norm_np = np.tile([0, 0, 1], (n_points, 1)).astype(np.float32)
    
    vol = SurfaceIrradianceVolume(pos_np.T, norm_np.T)
    
    # Point de requête aléatoire
    query_point = np.random.rand(3).astype(np.float32) * 10.0
    
    # Calcul via Dr.Jit
    found_idx = vol.nearest_point(mi.Point3f(query_point))
    
    # Calcul via Numpy (vérité terrain)
    dists = np.linalg.norm(pos_np - query_point, axis=1)
    expected_idx = np.argmin(dists)
    
    assert int(found_idx[0]) == expected_idx

def test_nearest_point_exact():
    """Vérifie que si on cherche un point qui existe, on le trouve."""
    pos_np = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ], dtype=np.float32)
    norm_np = np.tile([0, 0, 1], (4, 1)).astype(np.float32)
    
    vol = SurfaceIrradianceVolume(pos_np.T, norm_np.T)
    
    # Requête sur le point à [1, 1, 0] (index 3)
    query_point = mi.Point3f(1.0, 1.0, 0.0)
    found_idx = vol.nearest_point(query_point)
    
    assert int(found_idx[0]) == 3

if __name__ == "__main__":
    pytest.main([__file__])
