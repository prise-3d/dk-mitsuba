import mitsuba as mi
# Configure Mitsuba variant for tests
mi.set_variant('llvm_ad_rgb')

import drjit as dr
import numpy as np
import os
import pytest
from local_irradiance import SurfaceIrradianceVolume



def test_initialization():
    """
    check that the structure is initialized with the correct dimensions.
    """
    n_points = 5
    positions = np.zeros((3, n_points), dtype=np.float32)
    normals = np.zeros((3, n_points), dtype=np.float32)
    normals[2, :] = 1.0 # Toutes les normales vers Z+
    
    res_u, res_v = 4, 8
    grid_res = 16
    scene = mi.load_dict({'type': 'scene', 's': {'type': 'sphere'}})

    vol = SurfaceIrradianceVolume(scene, positions, normals, res_u, res_v, grid_res)
    
    assert vol.n_points == n_points
    assert vol.res_u == res_u
    assert vol.res_v == res_v
    assert vol.n_bins_per_point == res_u * res_v
    assert dr.width(vol.sum_r) == n_points * res_u * res_v
    assert dr.all(vol.visit_counts == 0)[0]

def test_update_and_query_averaging():
    """
    Check that updates correctly accumulate and average the irradiance values.
    """
    
    positions = mi.Point3f([0], [0], [0])
    normals = mi.Vector3f([0], [0], [1])
    scene = mi.load_dict({'type': 'scene', 's': {'type': 'sphere'}})
    vol = SurfaceIrradianceVolume(scene, positions, normals, 2, 2, 16)
    
    idx = mi.UInt32([0])
    direction = mi.Vector3f([0], [0], [1])
    active = mi.Bool([True])
    
    # first update
    vol.update(idx, direction, mi.Color3f(10.0), active)
    q_data = vol.get_q_data(idx)
    # For normal direction (0,0,1) and 2x2 resolution, flat_idx calculation results in bin 2
    assert dr.allclose(q_data[2], 10.0)
    
    # second update (same bin)
    vol.update(idx, direction, mi.Color3f(20.0), active)
    q_data = vol.get_q_data(idx)
    assert dr.allclose(q_data[2], 15.0) # (10 + 20) / 2

def test_directional_bins():
    """Check that different directions fall into different bins."""
    positions = mi.Point3f([0], [0], [0])
    normals = mi.Vector3f([0], [0], [1])
    scene = mi.load_dict({'type': 'scene', 's': {'type': 'sphere'}})
    vol = SurfaceIrradianceVolume(scene, positions, normals, 4, 4)
    
    idx = mi.UInt32([0])
    dir_up = mi.Vector3f([0, 0, 1])
    dir_side = mi.Vector3f([1, 0, 0])
    active = mi.Bool([True])
    
    vol.update(idx, dir_up, mi.Color3f(100.0), active)
    
    q_data = vol.get_q_data(idx)
    # 4x4 bins: dir_up (0,0,1) maps to bin 12, dir_side (1,0,0) maps to bin 0
    q_up = mi.luminance(q_data[12])
    q_side = mi.luminance(q_data[0])
    
    assert dr.allclose(q_up, 100.0)
    assert dr.allclose(q_side, 0.0)

def test_nearest_point():
    """Check that nearest_point finds the closest point (comparison with numpy)."""
    n_points = 10
    # We generate points in [-1, 1] to match the BBox of the default sphere
    pos_np = (np.random.rand(n_points, 3).astype(np.float32) * 2.0 - 1.0)
    norm_np = np.tile([0, 0, 1], (n_points, 1)).astype(np.float32)
    
    scene = mi.load_dict({'type': 'scene', 's': {'type': 'sphere'}})
    # We increase grid_res to minimize discretization error during the test
    vol = SurfaceIrradianceVolume(scene, pos_np.T, norm_np.T, grid_res=32)
    
    # We query a random point and check that the found point is close to the expected nearest point
    query_point = (np.random.rand(3).astype(np.float32) * 2.0 - 1.0)
    
    # Compute via Dr.Jit
    found_idx = vol.nearest_point(mi.Point3f(query_point), mi.Vector3f(0, 0, 1))
    
    # Calculate via Numpy (truth)
    dists = np.linalg.norm(pos_np - query_point, axis=1)
    expected_idx = np.argmin(dists)
        
    # check that the found point is within a reasonable distance of the expected nearest point (considering grid discretization)
    dist_found = np.linalg.norm(pos_np[found_idx] - query_point)
    dist_expected = dists[expected_idx]
    
    assert dist_found <= dist_expected + 0.2  # error margin due to the grid (?)

def test_nearest_point_exact():
    """Check that if we search for a point that exists, we find it."""
    pos_np = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ], dtype=np.float32)
    norm_np = np.tile([0, 0, 1], (4, 1)).astype(np.float32)
    
    scene = mi.load_dict({'type': 'scene', 's': {'type': 'sphere'}})
    vol = SurfaceIrradianceVolume(scene, pos_np.T, norm_np.T)
    
    # Requête sur le point à [1, 1, 0] (index 3)
    query_point = mi.Point3f(1.0, 1.0, 0.0)
    found_idx = vol.nearest_point(query_point, mi.Vector3f(0, 0, 1))
    
    assert int(found_idx[0]) == 3

# test sample_points_on_scene
def test_sample_points_on_scene():
    """Check that the sampling function generates the correct number of points and that the data is consistent."""
    scene = mi.load_dict({'type': 'scene', 's': {'type': 'sphere'}})
    n_points = 50
    vol = SurfaceIrradianceVolume.from_scene(scene, n_points)
    
    assert dr.width(vol.positions) > 0
    assert dr.width(vol.normals) > 0
    
    # Check that the normals are unit vectors
    norms = np.linalg.norm(np.array(vol.normals), axis=0)
    assert np.allclose(norms, 1.0)

# test save function in DistrubuteSurfacePointsonScene
def test_save_function():
    """Check that the save function creates a file with the correct data."""
    delete_files = False    
    scene_path='scenes/cbox/cbox.xml'
    scene = mi.load_file(scene_path)
    vol = SurfaceIrradianceVolume.from_scene(scene, 100)
    
    # Sauvegarder dans un fichier temporaire
    output_path = 'test_surface_points.ply'
    vol.save(output_path)
    

    assert os.path.exists(output_path)
    with open(output_path, 'r') as f:
        assert f.readline().strip() == "ply"

    output_path_hemisphere = 'hemisphere_points.ply'
    vol.save_hemi(output_path_hemisphere)
        

    if delete_files:
        os.remove(output_path)


if __name__ == "__main__":
    pytest.main([__file__])
