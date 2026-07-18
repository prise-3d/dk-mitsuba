"""
Reproduction of figure 8 from Dahm & Keller (2017),
"Learning Light Transport the Reinforced Way".

The paper plots the number of "valid" paths (paths connecting to a light
source, i.e. with non-zero contribution) per frame, as a function of the
number of accumulated frames, for two sampling strategies:
    - RL-based IS   : guiding by the learned Q-values (Expected Sarsa),
    - BRDF-based IS : classic importance sampling based on the BSDF alone.

As in the paper, one path is traced per pixel per frame, and Q-learning
happens online during rendering: the RL curve therefore shows the learning
progress over accumulated frames.

By default, next event estimation is disabled for both methods: with NEE,
almost every path would reach a light source through shadow rays and the
count would become trivial. The measurement therefore evaluates, as in the
paper, the ability of the directional sampling to reach the light. (Without
NEE the MIS weighting of this implementation underestimates radiance, but
this does not change whether a path contribution is zero or not.)

Usage:
    python plot_zero_contrib.py                       # cbox, 256x256, 400 frames
    python plot_zero_contrib.py --frames 1000         # as in the paper
    python plot_zero_contrib.py --scene scenes/cbox/cbox_occluded.xml
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')

import local_irradiance  # noqa: F401 -- registers 'rl_integrator'

# Validated categorical palette (slots 1 and 2, light mode)
COLOR_RL = '#2a78d6'    # blue : RL-based IS
COLOR_BRDF = '#1baf7a'  # aqua : BRDF-based IS
COLOR_TEXT = '#3a3a38'
COLOR_GRID = '#e6e5e1'


def load_scene(path, resx, resy):
    scene = mi.load_file(path)
    if resx and resy:
        params = mi.traverse(scene)
        for key in params.keys():
            if key.endswith('.film.size'):
                params[key] = [resx, resy]
                params.update()
                break
    return scene


def trace_frame(scene, integrator, sampler_proto, seed):
    """Traces one path per pixel and returns (n_valid, n_paths)."""
    sensor = scene.sensors()[0]
    size = sensor.film().crop_size()
    n = size.x * size.y

    sampler = sampler_proto.clone()
    sampler.seed(seed, n)

    idx = dr.arange(mi.UInt32, n)
    pos = mi.Vector2f(mi.Float(idx % size.x), mi.Float(idx // size.x))
    pos = (pos + sampler.next_2d()) / mi.Vector2f(mi.Float(size.x), mi.Float(size.y))
    ray, _ = sensor.sample_ray(0.0, sampler.next_1d(), pos, sampler.next_2d())

    active = dr.full(mi.Bool, True, n)
    result, _, _ = integrator.sample(scene, sampler, ray, None, active)

    lum = np.asarray(mi.luminance(result))
    return int((lum > 0).sum()), n


def run_experiment(scene, integrator, n_frames, base_seed, label):
    """Counts, frame by frame, the paths with non-zero contribution."""
    sampler_proto = mi.load_dict({'type': 'independent'})
    valid_counts = np.zeros(n_frames, dtype=np.int64)
    n_paths = 0
    t0 = time.perf_counter()
    for f in range(n_frames):
        valid_counts[f], n_paths = trace_frame(scene, integrator, sampler_proto, base_seed + f)
        if (f + 1) % 25 == 0 or f == n_frames - 1:
            print(  f"  [{label}] frame {f + 1}/{n_frames}: "
                    f"{valid_counts[f]}/{n_paths} valid paths "
                    f"({time.perf_counter() - t0:.1f}s)")
    return valid_counts, n_paths


def make_figure(valid_rl, valid_brdf, n_paths, out_path, meta=None):
    frames = np.arange(1, len(valid_rl) + 1)

    # Improvement factor over the last 10% of frames (the "43.49x"
    # annotation of figure 8)
    tail = max(1, len(frames) // 10)
    mean_rl, mean_brdf = valid_rl[-tail:].mean(), valid_brdf[-tail:].mean()
    ratio = mean_rl / max(mean_brdf, 1e-9)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    fig.patch.set_facecolor('white')

    ax.plot(frames, valid_rl, color=COLOR_RL, lw=2, label='RL-based IS')
    ax.plot(frames, valid_brdf, color=COLOR_BRDF, lw=2, label='BRDF-based IS')
    # Direct labels at the end of each curve (both series stay identifiable
    # without relying on color alone)
    for y, color, name in [(valid_rl, COLOR_RL, 'RL'), (valid_brdf, COLOR_BRDF, 'BRDF')]:
        ax.annotate(name, (frames[-1], y[-tail:].mean()),
                    xytext=(6, 0), textcoords='offset points',
                    color=color, fontsize=10, fontweight='bold', va='center')
    if meta:
        ax.set_title(meta, color=COLOR_TEXT, fontsize=9)
    ax.set_xlabel('accumulated frames', color=COLOR_TEXT)
    ax.set_ylabel('valid paths per frame', color=COLOR_TEXT)
    ax.set_xlim(0, frames[-1] * 1.06)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color=COLOR_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT)

    # Double-headed arrow between the two curves annotated with the
    # improvement factor, like the "43.49x" of figure 8
    i0 = int(len(frames) * 0.62)
    win = slice(max(0, i0 - tail // 2), i0 + tail // 2 + 1)
    y_hi, y_lo = valid_rl[win].mean(), valid_brdf[win].mean()
    ax.annotate('', xy=(frames[i0], y_hi), xytext=(frames[i0], y_lo),
                arrowprops=dict(arrowstyle='<->', color=COLOR_TEXT, lw=1.2))
    ax.annotate(f'{ratio:.2f}x', (frames[i0], (y_hi + y_lo) / 2),
                xytext=(8, 0), textcoords='offset points',
                color=COLOR_TEXT, fontsize=12, fontweight='bold', va='center')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend( handles, labels, frameon=False, ncol=2, loc='upper center',
                bbox_to_anchor=(0.5, 0.92), labelcolor=COLOR_TEXT)
    fig.suptitle(   'RL-based vs BRDF-based importance sampling',
                    color=COLOR_TEXT, fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Figure saved: {out_path}")
    return ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--scene', default='scenes/cbox/cbox.xml')
    parser.add_argument('--resx', type=int, default=0,
                        help='film width; with --resy, overrides --res (e.g. 1280x720 as in the paper)')
    parser.add_argument('--resy', type=int, default=0,
                        help='film height; with --resx, overrides --res')
    parser.add_argument('--res', type=int, default=256,
                        help='film resolution (0 = native scene resolution)')
    parser.add_argument('--frames', type=int, default=400,
                        help='number of accumulated frames (1000 in the paper)')
    parser.add_argument('--probes', type=int, default=4096,
                        help='number of spatial probes for learning')
    parser.add_argument('--grid-res', type=int, default=32,
                        help='resolution of the probe lookup grid (cells per axis)')
    parser.add_argument('--res-u', type=int, default=8,
                        help='directional bins in azimuth per probe')
    parser.add_argument('--res-v', type=int, default=8,
                        help='directional bins in elevation per probe')
    parser.add_argument('--q-init-value', type=float, default=1.0,
                        help='positive uniform Q initialization (radiance units)')
    parser.add_argument('--q-init-weight', type=float, default=8.0,
                        help='pseudo-visit weight of the Q initialization per bin')
    parser.add_argument('--refresh', choices=['frame', 'bounce'], default='frame',
                        help='rebuild guiding distributions once per frame (paper) or at every bounce')
    parser.add_argument('--max-depth', type=int, default=8,
                        help='path length cap (bounces) for both methods; the BRDF baseline '
                             'scales almost linearly with it')
    parser.add_argument('--grid-k', type=int, default=4,
                        help='candidate probes per grid cell for the normal-aware lookup')
    parser.add_argument('--nee', action='store_true',
                        help='enable next event estimation for both methods')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--metatitle', action='store_true',
                        help='add scene name, grid size, probe and candidate counts to the figure title')
    parser.add_argument('--out', default='zero_contrib_comparison.png')
    args = parser.parse_args()

    if args.resx and args.resy:
        scene = load_scene(args.scene, args.resx, args.resy)
    else:
        scene = load_scene(args.scene, args.res, args.res)
    size = scene.sensors()[0].film().crop_size()
    print(  f"Scene: {args.scene} ({size.x}x{size.y}, {args.frames} frames, "
            f"NEE {'on' if args.nee else 'off'})")

    # BRDF importance sampling alone (same integrator, guiding disabled)
    print("\n== BRDF-based IS ==")
    integ_brdf = mi.load_dict({ 'type': 'rl_integrator', 'enable_guiding': False,
                                'max_depth': args.max_depth})
    integ_brdf.next_event_estimation = args.nee
    valid_brdf, n_paths = run_experiment(scene, integ_brdf, args.frames, args.seed, 'BRDF')

    # RL guiding, online learning during the frames
    print("\n== RL-based IS ==")
    integ_rl = mi.load_dict({
        'type': 'rl_integrator',
        'enable_guiding': True,
        'update_q': True,
        'n_probes': args.probes,
        'resolution_u': args.res_u,
        'resolution_v': args.res_v,
        'q_init_value': args.q_init_value,
        'q_init_weight': args.q_init_weight,
        'grid_res': args.grid_res,
        'grid_k': args.grid_k,
        'refresh': args.refresh,
        'max_depth': args.max_depth,
    })
    integ_rl.next_event_estimation = args.nee
    valid_rl, _ = run_experiment(scene, integ_rl, args.frames, args.seed, 'RL')

    np.savez(args.out.rsplit('.', 1)[0] + '.npz',
             valid_rl=valid_rl, valid_brdf=valid_brdf,
             n_paths=n_paths, frames=args.frames,
             # run configuration, for comparing archived runs
             scene=args.scene, res=(size.x, size.y), probes=args.probes,
             grid_res=args.grid_res, grid_k=args.grid_k,
             nee=args.nee, seed=args.seed)

    meta = None
    if args.metatitle:
        import os
        meta = (f"{os.path.basename(args.scene)} | grid {args.grid_res}^3 | "
                f"{args.probes} probes | k={args.grid_k}")
    ratio = make_figure(valid_rl, valid_brdf, n_paths, args.out, meta=meta)
    print(  f"\nValid paths (mean over last 10% of frames): "
            f"RL {valid_rl[-max(1, args.frames // 10):].mean():.0f} vs "
            f"BRDF {valid_brdf[-max(1, args.frames // 10):].mean():.0f} "
            f"-> {ratio:.2f}x improvement over {n_paths} paths/frame")


if __name__ == '__main__':
    main()
