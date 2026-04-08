import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant('llvm_ad_rgb')

def check_normals():
    scene = mi.load_dict({
        "type": "scene",
        "sphere": {"type": "sphere", "radius": 1.0}
    })
    from local_irradiance import SurfaceIrradianceVolume
    vol = SurfaceIrradianceVolume.from_scene(scene, 100)
    
    pos = np.array(vol.positions)
    norm = np.array(vol.normals)
    
    # For a sphere at (0,0,0), dot(p, n) should be positive (1.0)
    dots = np.sum(pos * norm, axis=0)
    print(f"Mean dot(p, n) for sphere: {np.mean(dots)}")
    if np.mean(dots) > 0:
        print("Normals point OUTWARD (Correct for mitsuba spheres)")
    else:
        print("Normals point INWARD")

if __name__ == "__main__":
    check_normals()
