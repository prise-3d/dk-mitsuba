import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import os

mi.set_variant('llvm_ad_rgb')
from local_irradiance import RLIntegrator

def compare_integrators(scene_path='scenes/cbox/cbox.xml', spp=256):
    scene = mi.load_file(scene_path)
    
    # 1. Mitsuba's standard Path Tracer
    path_integrator = mi.load_dict({'type': 'path'})
    img_ref = mi.render(scene, integrator=path_integrator, spp=spp)
    mi.util.write_bitmap('compare_ref.png', img_ref)
    
    # 2. RLIntegrator with guiding=False
    rl_integrator = mi.load_dict({
        'type': 'rl_integrator',
        'enable_guiding': False,
        'n_probes': 100
    })
    img_rl = mi.render(scene, integrator=rl_integrator, spp=spp)
    mi.util.write_bitmap('compare_rl_no_guiding.png', img_rl)
    
    mean_ref = np.mean(img_ref)
    mean_rl = np.mean(img_rl)
    
    print(f"Reference Mean: {mean_ref:.6f}")
    print(f"RL (No Guiding) Mean: {mean_rl:.6f}")
    print(f"Ratio: {mean_rl/mean_ref:.4f}")

    # Plot the difference
    diff = np.abs(np.array(img_ref) - np.array(img_rl))
    plt.imshow(np.mean(diff, axis=2), cmap='hot')
    plt.colorbar()
    plt.title('Absolute Difference (Ref vs RL No-Guiding)')
    plt.savefig('compare_diff.png')

if __name__ == "__main__":
    compare_integrators()
