import mitsuba as mi
import drjit as dr
import numpy as np
import pytest
import os

# Ensure we are in a supported variant
mi.set_variant('llvm_ad_rgb')

from local_irradiance import RLIntegrator

def calculate_mse(img1, img2):
    return np.mean((np.array(img1) - np.array(img2))**2)

@pytest.fixture
def scene():
    scene_path = 'scenes/cbox/cbox.xml'
    if not os.path.exists(scene_path):
        pytest.skip("Scene file not found")
    scene = mi.load_file(scene_path)
    # Very low res for fast tests
    res = 256
    params = mi.traverse(scene)
    for key in params.keys():
        if key.endswith('.film.size'):
            params[key] = [res, res]
            params.update()
            break
    return scene

def test_learning_improvement(scene):
    """
    Verifies if RL guiding reduces the error compared to classic Path Tracing.
    MSE(Guided, Ref) < MSE(NoGuiding, Ref)
    """
    # Récupérer la liste des formes de la scène
    shapes = scene.shapes()

    # Boucler sur les formes et sauvegarder celles qui sont des maillages
    for i, shape in enumerate(shapes):
        # On vérifie si la shape a des données de maillage (vertices/faces)
        if isinstance(shape, mi.Mesh):
            filename = f"shape_{i}_{shape.id()}.ply"
            shape.write_ply(filename)
            print(f"Sauvegardé : {filename}")
        else:
            print(f"La shape {i} n'est pas un maillage (ex: sphère analytique), sautée.")

    print("\n=== Testing RL Guiding Improvement ===")
    spp_test = 64
    
    # Reference (Ground Truth) - Moderate SPP Path Tracing
    print("\nRendering Reference (256 spp)...")
    ref_integrator = mi.load_dict({"type": "path"})    

    img_ref = mi.render(scene, integrator=ref_integrator, spp=256, seed=0)
    
    # No Guiding - budget spp_test
    print(f"Rendering No Guiding ({spp_test} spp)...")
    integrator_no_guiding = mi.load_dict({
        "type": "rl_integrator",
        "enable_guiding": False
    })
    img_no_guiding = mi.render(scene, integrator=integrator_no_guiding, spp=spp_test, seed=1)
    
    # Guided RL - same spp_test budget, but with guiding enabled
    print(f"Training and Rendering Guided RL ({spp_test} spp)...")
    integrator_guided = mi.load_dict({
        "type": "rl_integrator",
        "enable_guiding": True,
        "update_q": True,
        "n_probes": 200,
        "resolution_u": 8,
        "resolution_v": 8
    })
    
    # Training passes (some passes to fill Q-values)
    for i in range(5):
        mi.render(scene, integrator=integrator_guided, spp=4, seed=i+10)
    
    # Final render for measurement
    img_guided = mi.render(scene, integrator=integrator_guided, spp=spp_test, seed=1)

    integrator_guided.save_hemi_q_values('learned_q_values.ply')    
    

    mse_no_guiding = calculate_mse(img_no_guiding, img_ref)
    mse_guided = calculate_mse(img_guided, img_ref)
    
    print(f"MSE No Guiding: {mse_no_guiding:.6f}")
    print(f"MSE Guided RL:  {mse_guided:.6f}")
    
    improvement = (mse_no_guiding - mse_guided) / mse_no_guiding * 100
    print(f"Improvement: {improvement:.2f}%")

    # save images for visual inspection (not required for the test, but useful for debugging)
    mi.util.convert_to_bitmap(img_ref).write('test_ref.png')
    mi.util.convert_to_bitmap(img_no_guiding).write('test_no_guiding.png')
    mi.util.convert_to_bitmap(img_guided).write('test_guided.png')

    # for validation save the 3d scene into a ply file
    

    # According to user, this might fail currently
    assert mse_guided < mse_no_guiding

if __name__ == "__main__":
    pytest.main([__file__])
