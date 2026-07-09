"""
Compare the old C++-executable-based polytope calculation (Microstructure /
calculate_polytopes) against the new pure-Python/numba port (calculate_polytopes_python)
on a single 2D binary image, to verify the new code reproduces the old results.

Usage:
    python compare_polytopes.py --path_input test_images\\XCT_11.4um_binary0000.tif ^
        --cpathPn cpp_poly\\512\\Cpp_source\\Polytope ^
        --runtimePn cpp_poly\\512\\runtime ^
        --outputPn cpp_poly\\512\\runtime\\output

Requires the compiled Sample_Pn_UU executable to actually run (see README) - if it fails
with a missing-DLL error, this script will report that clearly for whichever polytopes
were requested instead of silently comparing against stale data.
"""
import argparse
import numpy as np
import tifffile

from src.micro_gui.analysis.smds import calculate_polytopes, calculate_polytopes_python

POLYTOPES = ['p3h', 'p3v', 'p4', 'p6', 'L']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input', required=True, type=str, help='Full path to the 2D tif image.')
    parser.add_argument('--cpathPn', required=True, type=str, help='Path to Polytope folder (old cpp method).')
    parser.add_argument('--runtimePn', required=True, type=str, help='Path to runtime folder (old cpp method).')
    parser.add_argument('--outputPn', required=True, type=str, help='Path to runtime/output folder (old cpp method).')
    parser.add_argument('--polytopes', nargs='+', default=POLYTOPES, help='Which polytopes to compare.')
    return parser.parse_args()


def main():
    args = parse_args()

    img = tifffile.imread(args.path_input).astype(np.uint8)
    print(f'Image: {args.path_input}, shape: {img.shape}, foreground fraction: {img.mean():.6f}')

    par = {'name': 'polytopes', 'begx': 0, 'begy': 0, 'nsamp': img.shape[0], 'edge_buffer': 0}
    cpathPn = args.cpathPn.rstrip('/\\') + '/'
    runtimePn = args.runtimePn.rstrip('/\\') + '/'
    outputPn = args.outputPn.rstrip('/\\') + '/'

    # new pure-Python port: computes everything requested in one call
    _, py_scaled = None, None
    py_raw, py_scaled = calculate_polytopes_python(img, polytopes=tuple(args.polytopes))

    print()
    print(f"{'polytope':10s} {'status':10s} {'max_abs_diff':>14s} {'max_rel_diff':>14s} {'n_mismatch(>1e-4)':>20s}")
    for name in args.polytopes:
        try:
            cpp_raw, _ = calculate_polytopes(img, par, outputPn, cpathPn, runtimePn, polytope=name)
        except Exception as e:
            print(f"{name:10s} {'CPP FAILED':10s}  {type(e).__name__}: {e}")
            continue

        mine = py_raw[name][:, 1]
        ref = cpp_raw[:, 1]
        diff = np.abs(mine - ref)
        rel = diff / np.maximum(np.abs(ref), 1e-12)
        n_mismatch = int(np.sum(diff > 1e-4))
        print(f"{name:10s} {'ok':10s} {diff.max():14.8f} {rel.max():14.8f} {n_mismatch:20d}")
        if n_mismatch > 0:
            bad_idx = np.where(diff > 1e-4)[0][:10]
            for i in bad_idx:
                print(f"    r={i}: python={mine[i]:.6f} cpp={ref[i]:.6f}")


if __name__ == '__main__':
    main()
