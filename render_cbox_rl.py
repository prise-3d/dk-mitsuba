import mitsuba as mi
import drjit as dr
import os

# Ensure the correct variant is set before importing the integrator
mi.set_variant('llvm_ad_rgb')

from local_irradiance import RLIntegrator

class CornellBoxRenderer:
    """
    A class to handle rendering the Cornell Box using the reinforcement learning integrator.
    """
    def __init__(self, scene_path='scenes/cbox/cbox.xml', n_probes=1000):
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")
        
        self.scene_path = scene_path
        self.n_probes = n_probes
        self.scene = mi.load_file(self.scene_path)
        
    def render(self, spp=1024, guiding=True, update_q=True, output_filename='cbox_rl.png'):
        """
        Renders the scene and saves the result.
        """
        print(f"Starting render with RLIntegrator ({self.n_probes} probes, {spp} spp)...")
        
        # Change resolution dynamically
        res = 256
        params = mi.traverse(self.scene)
        for key in params.keys():
            if key.endswith('.film.size'):
                params[key] = [res, res]
                params.update()
                break

        # Define the integrator dictionary
        integrator_dict = {
            "type": "rl_integrator",
            "n_probes": self.n_probes,
            "enable_guiding": guiding,
            "update_q": update_q
        }
        
        # Load the integrator
        integrator = mi.load_dict(integrator_dict)

        # Split SPP into multiple passes
        n_passes = 8
        spp_per_pass = max(1, spp // n_passes)
        
        image = None
        for i in range(n_passes):
            print(f"  Pass {i+1}/{n_passes} ({spp_per_pass} spp)...")
            pass_image = mi.render(self.scene, integrator=integrator, spp=spp_per_pass)
            if image is None:
                image = pass_image
            else:
                image = image + (pass_image - image) / (i + 1)
            
            if guiding and integrator.volume:
                stats = integrator.volume.get_stats()
                print(f"    Stats: visits={stats['total_visits']:.0f}, mean_q={stats['mean_q']:.4f}, max_q={stats['max_q']:.4f}")

        # Convert to sRGB and write to PNG
        bitmap = mi.util.convert_to_bitmap(image)
        bitmap.write(output_filename)
        print(f"Render completed. Image saved to: {output_filename}")
        
        return image
    
    def mi_render(self, integrator = mi.load_dict({"type": "path"}), spp=1024, res=256,
 output_filename='cbox_mi.png') :
        """
        Renders the scene using Mitsuba's default Path Tracer.
        """
        print(f"Starting render with Mitsuba's default Path Tracer ({spp} spp)...")

        # Change resolution dynamically
        params = mi.traverse(self.scene)
        for key in params.keys():
            if key.endswith('.film.size'):
                params[key] = [res, res]
                params.update()
                break

        ref_integrator = mi.load_dict({"type": "path"})
        img_ref = mi.render(self.scene, integrator=ref_integrator, spp=spp)
        bitmap = mi.util.convert_to_bitmap(img_ref)
        bitmap.write(output_filename)
        print(f"Render completed. Image saved to: {output_filename}")

    

if __name__ == "__main__":
    # Create the renderer instance
    renderer = CornellBoxRenderer(n_probes=1000)

    # Run the rendering process
    spp = 512

    print("--- Rendering with Guided RL (Updating Q) ---")
    renderer.render(spp=spp, output_filename='render_result.png')
    
    #print("\n--- Rendering with Guided RL (No Update) ---")
    #renderer.render(spp=spp, update_q=False, output_filename='render_result_no_update.png')
    
    #print("\n--- Rendering with No Guiding ---")
    #renderer.render(spp=spp, guiding=False, output_filename='render_result_no_guiding.png')

    print("\n--- Render with Mitsuba's default Path Tracer ---")
    renderer.mi_render(spp=spp, output_filename='render_result_mi.png')
    