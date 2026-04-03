import mitsuba as mi
import drjit as dr
import pytest
import numpy as np
import os

# Ensure we are in a supported variant
mi.set_variant('llvm_ad_rgb')

from local_irradiance import RLIntegrator

@pytest.fixture(scope="module")
def scene():
    scene_path = 'scenes/cbox/cbox.xml'
    if not os.path.exists(scene_path):
        pytest.skip("Scene file not found")
    scene = mi.load_file(scene_path)
    
    # Set a small resolution for tests
    res = 32
    params = mi.traverse(scene)
    for key in params.keys():
        if key.endswith('.film.size'):
            params[key] = [res, res]
            params.update()
            break
    return scene

def test_no_update_equals_no_guiding(scene):
    """
    Verifies that 'Guided (with update_q=False)' is statistically close to 'No Guiding' 
    when starting with zero Q-values.
    """
    spp = 64
    seed = 42

    # 1. Render without guiding
    integrator_dict_off = {
        "type": "rl_integrator",
        "enable_guiding": False,
        "n_probes": 100
    }
    integrator_off = mi.load_dict(integrator_dict_off)
    img_off = mi.render(scene, integrator=integrator_off, spp=spp, seed=seed)
    
    # 2. Render with guiding enabled but no update (so Q remains 0)
    integrator_dict_on_no_up = {
        "type": "rl_integrator",
        "enable_guiding": True,
        "update_q": False,
        "n_probes": 100
    }
    integrator_on_no_up = mi.load_dict(integrator_dict_on_no_up)
    img_on_no_up = mi.render(scene, integrator=integrator_on_no_up, spp=spp, seed=seed)

    mean_off = np.mean(img_off)
    mean_on_no_up = np.mean(img_on_no_up)
    
    print(f"\nMean Off: {mean_off:.6f}")
    print(f"Mean On (No Update): {mean_on_no_up:.6f}")
    
    # We use a relaxed tolerance because different code paths consume 
    # different amounts of random numbers, causing MC variance.
    # 1e-2 is enough to catch systematic energy bugs while ignoring noise.
    assert np.allclose(mean_off, mean_on_no_up, rtol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__])
