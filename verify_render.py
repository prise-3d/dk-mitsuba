import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant('llvm_ad_rgb')

def verify_render(filename='render_result.exr'):
    try:
        bitmap = mi.Bitmap(filename)
        data = np.array(bitmap)
        avg_val = np.mean(data)
        max_val = np.max(data)
        print(f"Verification of {filename}:")
        print(f"  Average pixel value: {avg_val:.6f}")
        print(f"  Max pixel value: {max_val:.6f}")
        
        if avg_val > 0:
            print("  SUCCESS: Image contains light.")
        else:
            print("  FAILURE: Image is completely black.")
            
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    verify_render()
