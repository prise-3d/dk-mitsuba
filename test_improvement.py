import mitsuba as mi
import drjit as dr
import numpy as np
import pytest
import os
import time

# Ensure we are in a supported variant
mi.set_variant('llvm_ad_rgb')

from local_irradiance import RLIntegrator

def calculate_mse(img1, img2):
    return np.mean((np.array(img1) - np.array(img2))**2)

def calculate_snr(img):
    signal_power = np.mean(np.array(img)**2)
    noise_power = np.mean((np.array(img) - np.mean(img))**2)
    return 10 * np.log10(signal_power / noise_power)

@pytest.fixture(scope="module")
def scene():
    scene_path = 'scenes/corridor/corridor_4.2.xml'
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

@pytest.fixture(scope="module")
def rendering_results(scene):
    """
    Common setup that performs all renders and calculates MSE values.
    Scope is 'module' to avoid re-rendering for each test method.
    """

    # Create output directory for PLY files
    os.makedirs('ply', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    # Récupérer la liste des formes de la scène
    shapes = scene.shapes()

    # Boucler sur les formes et sauvegarder celles qui sont des maillages
    for i, shape in enumerate(shapes):
        # On vérifie si la shape a des données de maillage (vertices/faces)
        if isinstance(shape, mi.Mesh):
            s_id = shape.id() if shape.id() else "unnamed"
            filename = f"ply/shape_{i}_{s_id}.ply"
            shape.write_ply(filename)
            print(f"Sauvegardé : {filename}")
        else:
            print(f"La shape {i} n'est pas un maillage (ex: sphère analytique), sautée.")

    print("\n=== Testing RL Guiding Improvement ===")
    spp_test = 64
    
    # Reference (Ground Truth) - Moderate SPP Path Tracing
    print("\nRendering Reference (256 spp)...")
    ref_integrator = mi.load_dict({"type": "path"})    

    start_time = time.perf_counter()
    img_ref = mi.render(scene, integrator=ref_integrator, spp=256, seed=0)
    ref_time = time.perf_counter() - start_time
    mi.util.convert_to_bitmap(img_ref).write('test/test_ref.png')
    print("Reference saved to test/test_ref.png")
    
    # No Guiding - budget spp_test
    print(f"Rendering No Guiding ({spp_test} spp)...")
    integrator_no_guiding = mi.load_dict({
        "type": "rl_integrator",
        "enable_guiding": False
    })
    start_time = time.perf_counter()
    img_no_guiding = mi.render(scene, integrator=integrator_no_guiding, spp=spp_test, seed=1)
    no_guiding_time = time.perf_counter() - start_time
    mi.util.convert_to_bitmap(img_no_guiding).write('test/test_no_guiding.png')
    print(f"No Guiding saved to test/test_no_guiding.png")
    
    # Guided RL - same spp_test budget, but with guiding enabled
    print(f"Training and Rendering Guided RL ({spp_test} spp)...")
    integrator_guided = mi.load_dict({
        "type": "rl_integrator",
        "enable_guiding": True,
        "update_q": True,
        "n_probes": 8192,
        "resolution_u": 8,
        "resolution_v": 8
    })
    
    # Training passes (some passes to fill Q-values)
    start_time = time.perf_counter()
    for i in range(16):  # 16 passes with 4 spp each = 64 spp total for training
        mi.render(scene, integrator=integrator_guided, spp=4, seed=i+10)
    training_time = time.perf_counter() - start_time
    
    # Final render for measurement
    start_time = time.perf_counter()
    img_guided = mi.render(scene, integrator=integrator_guided, spp=spp_test, seed=1)
    guided_time = time.perf_counter() - start_time
    mi.util.convert_to_bitmap(img_guided).write('test/test_guided.png')
    print(f"Guided RL saved to test/test_guided.png")

    integrator_guided.save_hemi_q_values('ply/learned_q_values.ply')    
    
    mse_no_guiding = calculate_mse(img_no_guiding, img_ref)
    mse_guided = calculate_mse(img_guided, img_ref)

    snr_guided = calculate_snr(img_guided)
    snr_no_guiding = calculate_snr(img_no_guiding)

    
    
    print(f"\n--- Performance Summary ---")
    print(f"Reference Time (256 spp): {ref_time:7.2f}s")
    print(f"No Guiding Time ({spp_test} spp): {no_guiding_time:7.2f}s")
    print(f"RL Training Time (5x4 spp):   {training_time:7.2f}s")
    print(f"RL Guided Render Time ({spp_test} spp): {guided_time:7.2f}s")
    print(f"Overhead Ratio (Guided/None): {guided_time / no_guiding_time:7.2f}x")

    print(f"\n--- Quality Summary ---")
    print(f"MSE No Guiding (to ref): {mse_no_guiding:.6f}")
    print(f"MSE Guided RL (to ref):  {mse_guided:.6f}")
    print(f"SNR No Guiding: {snr_no_guiding:.2f} dB")
    print(f"SNR Guided RL:  {snr_guided:.2f} dB")

    improvement = (mse_no_guiding - mse_guided) / mse_no_guiding * 100
    print(f"Improvement: {improvement:.2f}%")
   
    return {
        "mse_no_guiding": mse_no_guiding,
        "mse_guided": mse_guided,
        "snr_no_guiding": snr_no_guiding,
        "snr_guided": snr_guided
    }


def test_snr_improvement(rendering_results):
    assert rendering_results["snr_guided"] > rendering_results["snr_no_guiding"], "Guided RL should have higher SNR than No Guiding"

def test_improvement_vs_reference(rendering_results):
    assert rendering_results["mse_guided"] < rendering_results["mse_no_guiding"], "Guided RL should have lower MSE than No Guiding"


if __name__ == "__main__":
    pytest.main([__file__])
