import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

def test_bsdf_coords():
    scene = mi.load_dict({
        "type": "scene",
        "sphere": {"type": "sphere", "bsdf": {"type": "diffuse"}}
    })
    sphere = scene.shapes()[0]
    bsdf = sphere.bsdf()
    
    # Create a fake SurfaceInteraction
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.p = [0, 0, 0]
    si.n = [0, 0, 1]
    si.sh_frame = mi.Frame3f(si.n)
    si.wi = [0, 0, 1] # incident from above
    
    ctx = mi.BSDFContext()
    
    # Test direction in world space (same as local here)
    wo_world = mi.Vector3f(0, 0, 1)
    val = bsdf.eval(ctx, si, wo_world)
    pdf = bsdf.pdf(ctx, si, wo_world)
    print(f"World (0,0,1): eval={val}, pdf={pdf}")

    # Test direction in world space (different from local)
    si.n = mi.Vector3f(1, 0, 0)
    si.sh_frame = mi.Frame3f(si.n)
    # local wo = (0,0,1) would be world wo = (1,0,0)
    wo_world = mi.Vector3f(1, 0, 0)
    val = bsdf.eval(ctx, si, wo_world)
    pdf = bsdf.pdf(ctx, si, wo_world)
    print(f"World (1,0,0) with normal (1,0,0): eval={val}, pdf={pdf}")
    
    # Test bsdf.sample coordinate system
    ps = mi.Point2f(0.5, 0.5)
    si.n = mi.Vector3f(1, 0, 0)
    si.sh_frame = mi.Frame3f(si.n)
    bs_sample, bs_weight = bsdf.sample(ctx, si, 0.5, ps)
    print(f"Sample wo: {bs_sample.wo} with normal {si.n}")
    # If wo is world space, it should be something like (0.7, 0, 0.7) or similar.
    # If wo is local space, it should be something like (0, 0, 1) or similar.

if __name__ == "__main__":
    import drjit as dr
    test_bsdf_coords()
