"""Parallel high-resolution render of the Julia-dimension parameter map.

Computes dim J_c (Bowen pressure formula, see mandel.pressure_dimension) on a
P x P grid over mandel.MANDEL_EXTENT using a process pool -- the computation
is embarrassingly parallel across parameters, and numpy's elementwise work is
single-threaded, so worker processes scale near-linearly.

Outputs, in the working directory:
    dimension_map_<P>.npy          raw (P, P) float array of dimensions
    dimension_map_<P>.png          pixel-exact image (one pixel per parameter)
    dimension_map_<P>_labeled.png  figure with axes and colorbar

Run:  py render_dimension.py
"""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import mandel

P = 2400          # parameters per side of the map
DEPTH = 14        # preimage-tree depth (matches mandel.DIM_HIRES accuracy)
WORKERS = 16      # of 20 cores; leaves headroom for the desktop
ROWS_PER_TASK = 4


def compute_rows(r0):
    """Compute map rows [r0, r0 + ROWS_PER_TASK) of the P x P grid.

    Rebuilds the needed grid rows locally from the extent (cheaper than
    pickling grid slices to every worker) and sub-chunks the parameters so
    each pressure_dimension call stays within mandel.DIM_BATCH tree leaves.
    """
    xmin, xmax, ymin, ymax = mandel.MANDEL_EXTENT
    r1 = min(r0 + ROWS_PER_TASK, P)
    x = np.linspace(xmin, xmax, P)
    y = np.linspace(ymin, ymax, P)[r0:r1]
    cs = (x[None, :] + 1j * y[:, None]).ravel()
    out = np.empty(cs.size)
    step = max(1, mandel.DIM_BATCH // 2 ** DEPTH)
    for k in range(0, cs.size, step):
        out[k:k + step] = mandel.pressure_dimension(cs[k:k + step], DEPTH)
    return r0, out.reshape(r1 - r0, P)


def main():
    # Conjugation symmetry (dim J_c = dim J_conj(c)): MANDEL_EXTENT is
    # symmetric about the real axis, so only the top half is computed.
    start = P // 2
    tasks = list(range(start, P, ROWS_PER_TASK))
    dims = np.empty((P, P))
    t0 = time.perf_counter()
    done_rows = 0
    total_rows = P - start
    print(f'rendering {P}x{P} dimension map, depth {DEPTH}, '
          f'{WORKERS} workers, {len(tasks)} tasks', flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(compute_rows, r0) for r0 in tasks]
        for fut in as_completed(futures):
            r0, block = fut.result()
            dims[r0:r0 + block.shape[0]] = block
            done_rows += block.shape[0]
            elapsed = time.perf_counter() - t0
            eta = elapsed / done_rows * (total_rows - done_rows)
            if done_rows % (ROWS_PER_TASK * 25) == 0 or done_rows == total_rows:
                print(f'  {done_rows}/{total_rows} rows, {elapsed/60:.1f} min '
                      f'elapsed, ~{eta/60:.1f} min left', flush=True)
    dims[:start] = dims[::-1][:start].copy()   # mirror: row k <- P-1-k

    np.save(f'dimension_map_{P}.npy', dims)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imsave(f'dimension_map_{P}.png', dims, cmap=mandel.CMAP,
               origin='lower')
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(dims, origin='lower', extent=mandel.MANDEL_EXTENT,
                   cmap=mandel.CMAP)
    ax.set_xlabel('Re c')
    ax.set_ylabel('Im c')
    ax.set_title(f'Hausdorff dimension of $J_c$ (Bowen pressure formula, '
                 f'depth-{DEPTH} preimage trees, {P}x{P} parameters)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                 label='dim $J_c$')
    fig.savefig(f'dimension_map_{P}_labeled.png', dpi=300,
                bbox_inches='tight')
    print(f'saved dimension_map_{P}.npy / .png / _labeled.png '
          f'in {(time.perf_counter()-t0)/60:.1f} min '
          f'(range {dims.min():.3f} .. {dims.max():.3f})', flush=True)


if __name__ == '__main__':
    main()
