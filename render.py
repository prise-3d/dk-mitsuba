"""
Rendering of RL-guided and BRDF path tracing.

Two usages:

1. Single render, saved as linear float32 OpenEXR for visual inspection:
        python render.py --mode rl   --spp 64
        python render.py --mode brdf --spp 64

2. Equal-time comparison (--mode compare): both methods accumulate render
    passes of samples each until the same wall-clock budget is exhausted;
    the final image is the average of the passes. The sample count
    each method reaches within the budget is reported on the figure. The RL
    integrator learns online during the timed passes, so its training cost is
    included in its budget. A 1 spp warm-up pass, excluded from the budget,
    absorbs JIT compilation for both methods.
        python render.py --mode compare --budget 60

NEE is enabled by default (realistic renders); disable with --no-nee.
An optional path-traced reference (--ref-spp N) adds MSE/relMSE figures.

example:

uv run render.py --mode compare \
    --scene scenes/corridor.xml \
    --resx 480 --resy 480 \
    --budget 60 \
    --pass-spp 4 \
    --no-nee \
    --ref-spp 2048 \
    --out-prefix corridor-cuda
"""

import argparse
import os
import time

import numpy as np
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
# mi.set_variant('llvm_ad_rgb')

import local_irradiance  # noqa: F401 -- registers 'rl_integrator'


def load_scene(args):
    kwargs = {}
    if args.resx and args.resy:
        kwargs = {'resx': args.resx, 'resy': args.resy}
        try:
            return mi.load_file(args.scene, **kwargs)
        except RuntimeError:
            pass  # scene without resx/resy defaults: fall through and patch the film
    scene = mi.load_file(args.scene)
    if args.resx and args.resy:
        params = mi.traverse(scene)
        for key in params.keys():
            if key.endswith('.film.size'):
                params[key] = [args.resx, args.resy]
                params.update()
                break
    return scene


def make_integrator(mode, args):
    integ = mi.load_dict({
        'type': 'rl_integrator',
        'enable_guiding': mode == 'rl',
        'update_q': mode == 'rl',
        'n_probes': args.probes,
        'grid_res': args.grid_res,
        'grid_k': args.grid_k,
        'q_init_value': args.q_init_value,
        'q_init_weight': args.q_init_weight,
    })
    integ.next_event_estimation = not args.no_nee
    return integ


def save_image(img, path):
    """Writes the image as linear float32 OpenEXR."""
    mi.Bitmap(np.asarray(img, dtype=np.float32)).write(path)
    print(f"saved {path}")


def render_single(scene, mode, args):
    integ = make_integrator(mode, args)
    mi.render(scene, integrator=integ, spp=1, seed=12345)  # JIT warm-up
    t0 = time.perf_counter()
    img = mi.render(scene, integrator=integ, spp=args.spp, seed=args.seed)
    elapsed = time.perf_counter() - t0
    print(f"[{mode}] {args.spp} spp in {elapsed:.1f}s")
    return np.array(img), elapsed


def render_equal_time(scene, mode, args):
    """Accumulates --pass-spp passes until the time budget is exhausted."""
    integ = make_integrator(mode, args)
    mi.render(scene, integrator=integ, spp=1, seed=12345)  # JIT warm-up
    acc, n, t0 = None, 0, time.perf_counter()
    while time.perf_counter() - t0 < args.budget or n == 0:
        img = np.array(mi.render(   scene, integrator=integ, spp=args.pass_spp,
                                    seed=args.seed + n))
        acc = img if acc is None else acc + img
        n += 1
    elapsed = time.perf_counter() - t0
    spp = n * args.pass_spp
    print(f"[{mode}] {spp} spp ({n} passes) in {elapsed:.1f}s")
    return acc / n, spp, elapsed


def mse(img, ref):
    d = img - ref
    return float(np.mean(d * d))


def rel_mse(img, ref):
    d = img - ref
    return float(np.mean(d * d / (ref * ref + 1e-2)))


def tonemap(img):
    return np.array(mi.util.convert_to_bitmap(img)) / 255.0


def side_by_side(images, labels, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 5), dpi=130)
    for ax, img, label in zip(np.atleast_1d(axes), images, labels):
        ax.imshow(tonemap(img))
        ax.set_title(label, fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--mode', choices=['brdf', 'rl', 'compare'], default='compare')
    parser.add_argument('--scene', default='scenes/cbox/cbox.xml')
    parser.add_argument('--resx', type=int, default=0, help='film width (0 = scene default)')
    parser.add_argument('--resy', type=int, default=0, help='film height (0 = scene default)')
    parser.add_argument('--spp', type=int, default=64, help='samples per pixel (single-render modes)')
    parser.add_argument('--budget', type=float, default=60.0,
                        help='wall-clock budget in seconds per method (compare mode)')
    parser.add_argument('--pass-spp', type=int, default=4,
                        help='samples per pixel per accumulation pass (compare mode)')
    parser.add_argument('--probes', type=int, default=4096)
    parser.add_argument('--grid-res', type=int, default=32)
    parser.add_argument('--grid-k', type=int, default=4)
    parser.add_argument('--q-init-value', type=float, default=1.0)
    parser.add_argument('--q-init-weight', type=float, default=8.0)
    parser.add_argument('--no-nee', action='store_true',
                        help='disable next event estimation for both methods')
    parser.add_argument('--ref-spp', type=int, default=0,
                        help='if > 0, render a path-traced reference at this spp and report MSE')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-prefix', default='render',
                        help='output file prefix (e.g. render_rl, render_brdf')
    args = parser.parse_args()

    scene = load_scene(args)
    size = scene.sensors()[0].film().crop_size()
    print(f"Scene: {args.scene} ({size.x}x{size.y}, NEE {'off' if args.no_nee else 'on'})")

    ref = None
    if args.ref_spp > 0:
        print(f"Rendering path-traced reference ({args.ref_spp} spp)...")
        ref_integ = mi.load_dict({'type': 'path', 'max_depth': 8})
        ref = np.array(mi.render(scene, integrator=ref_integ, spp=args.ref_spp, seed=987))
        save_image(ref, f"{args.out_prefix}_ref.exr")

    if args.mode in ('brdf', 'rl'):
        img, _ = render_single(scene, args.mode, args)
        save_image(img, f"{args.out_prefix}_{args.mode}.exr")
        if ref is not None:
            print(f"MSE {mse(img, ref):.6f} | relMSE {rel_mse(img, ref):.4f}")
        return

    # compare mode: same wall-clock budget for both methods
    results = {}
    for mode in ('brdf', 'rl'):
        img, spp, elapsed = render_equal_time(scene, mode, args)
        save_image(img, f"{args.out_prefix}_{mode}.exr")
        label = f"{mode.upper()} -- {spp} spp in {elapsed:.0f}s"
        if ref is not None:
            label += f"\nMSE {mse(img, ref):.5f} | relMSE {rel_mse(img, ref):.4f}"
        results[mode] = (img, label)

    images = [results['brdf'][0], results['rl'][0]]
    labels = [results['brdf'][1], results['rl'][1]]
    if ref is not None:
        images.append(ref)
        labels.append(f"Reference (path, {args.ref_spp} spp)")
    side_by_side(images, labels, f"{args.out_prefix}_compare.png")


if __name__ == '__main__':
    main()