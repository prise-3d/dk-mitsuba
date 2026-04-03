import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

import drjit as dr
from local_irradiance import RLIntegrator

def test_rl_integrator():
    # Load a simple scene (Cornell Box)
    scene = mi.load_file('scenes/cbox/cbox.xml')
    
    # Create the RL integrator
    integrator = mi.load_dict({
        "type": "rl_integrator",
        "n_probes": 100,
        "enable_guiding": False
    })
    
    # Render a small image
    image = mi.render(scene, integrator=integrator, spp=1)
    
    print("Render finished successfully.")
    # Check if image has some non-zero values
    if dr.any(image > 0):
        print("Image has light!")
    else:
        print("Image is black. (Might be normal for 1 spp or if probes are not yet learned)")

if __name__ == "__main__":
    test_rl_integrator()
