import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.widgets import RectangleSelector

# Full (home) viewing windows in the complex plane.
# Mandelbrot lives in the c-plane (parameter plane), Julia in the z-plane (dynamical plane).
MANDEL_EXTENT = (-2.0, 1.0, -1.5, 1.5)   # (xmin, xmax, ymin, ymax)
JULIA_EXTENT = (-2.0, 2.0, -2.0, 2.0)

# Two quality levels: coarse for live feedback while dragging, fine for the final image
# once the mouse button is released.
COARSE = dict(n=200, max_iter=100)
FINE = dict(n=800, max_iter=400)

# Export quality for saved PNGs (the `e` key / save_panel()). n is the output
# image size in pixels per side; the larger iteration budget resolves the
# thin boundary filaments that a 3000-px grid can now actually display.
HIRES = dict(n=3000, max_iter=1000)

# Dimension mode ('b' key): heat map of the Hausdorff dimension of the Julia
# set J_c over parameter space, computed per parameter from Bowen's pressure
# formula on a depth-level backward-orbit tree (2**depth preimages of the
# repelling fixed point). Unlike pixel-based box counting, this needs no
# Julia rendering and stays accurate for the Cantor-dust Julia sets outside
# the Mandelbrot set, whose sub-pixel gap structure a rendering cannot see.
DIM_QUALITY = dict(p=120, depth=13)
DIM_HIRES = dict(p=240, depth=14)   # `e`-key export: finer grid, deeper trees

# Preimage-tree leaves processed per vectorized batch (parameters * 2**depth).
# Bigger batches amortize numpy overhead but cost memory (~50 bytes/leaf of
# transient state during tree construction).
DIM_BATCH = 1_500_000

# Minimum seconds between recomputations while dragging, so motion events don't pile up.
DRAG_THROTTLE = 0.05

# Bailout radius for the escape test. Any orbit with |z| > 2 provably diverges, so
# membership in the set is identical for any radius >= 2; using a larger radius just
# lets the orbit grow a little past the threshold, which makes the smooth-coloring
# formula below more accurate (less visible banding).
ESCAPE_RADIUS = 4.0

# Perceptually uniform sequential colormap: lightness increases monotonically
# (so escape "speed" stays ordered even for colorblind viewers or in grayscale),
# while the hue sweep black -> purple -> red -> yellow keeps it vibrant.
CMAP = 'inferno'

# Difference-mode colors (chosen outside inferno's black->red->yellow sweep so
# the classification regions can't be confused with escape shading).
COLOR_BOTH = (0.0, 0.0, 0.0)        # black: both orbits bounded  (M_z0 ∩ M)
COLOR_CRIT_ONLY = (0.23, 0.52, 1.0)  # blue:  only critical orbit  (M ∖ M_z0)
COLOR_Z0_ONLY = (0.0, 1.0, 0.4)      # green: only z0's orbit      (the "dust")

# Survival ("un-dustified") mode: initial iteration threshold T for the solid
# finite-time stability set S_T = {c : z0's orbit survives >= T iterations}.
LEVEL_T0 = 50


def complex_grid(extent, n):
    """Return an n x n grid of complex numbers covering `extent`."""
    x = np.linspace(extent[0], extent[1], n)
    y = np.linspace(extent[2], extent[3], n)
    X, Y = np.meshgrid(x, y)
    return X + 1j * Y


def escape_time(z0, c, max_iter):
    """Iterate z -> z**2 + c and record a *smooth* escape time for each point.

    Either `z0` or `c` may be a grid while the other is a scalar:
      - Mandelbrot set: c is the grid, z0 is a single initial condition.
      - Julia set:      z0 is the grid, c is a single parameter.

    Returns an array `mu` where mu[i,j] is the smooth (fractional) iteration
    count at which the orbit escaped, or 0 if it never escaped within max_iter
    iterations (treated as inside the set).

    Smooth coloring: for an escaped point, the integer count i is corrected by
    -log2(log|z_i|), the fractional "overshoot" past the bailout radius. This
    makes mu vary continuously across the plane instead of jumping in integer
    steps, which removes the banding that hides fine filament structure.
    """
    z0, c = np.broadcast_arrays(np.asarray(z0, dtype=complex),
                                np.asarray(c, dtype=complex))
    z = z0.copy()                      # broadcast views are read-only; copy makes z writable
    mu = np.zeros(z.shape)
    alive = np.ones(z.shape, dtype=bool)   # points whose orbit hasn't escaped yet

    for i in range(1, max_iter + 1):
        z[alive] = z[alive] ** 2 + c[alive]
        escaped = alive & (np.abs(z) > ESCAPE_RADIUS)
        mu[escaped] = i + 1 - np.log2(np.log(np.abs(z[escaped])))
        alive &= ~escaped
        if not alive.any():
            break

    return mu


def shade(mu):
    """Compress smooth escape counts for display.

    log1p lifts the mid-range: points that survive many iterations (the ones
    tracing the fractal boundary) get spread across most of the colormap
    instead of being crushed into its top few percent. Interior points keep
    mu = 0, landing on the colormap's black end.
    """
    return np.log1p(mu)


def pressure_dimension(c_batch, depth, m=6):
    """Hausdorff dimension of J_c for each parameter, via Bowen's formula.

    For hyperbolic c (all of parameter space except the boundary of the
    Mandelbrot set and other non-hyperbolic parameters of measure zero, if
    any exist), dim J_c is the unique d at which the topological pressure of
    -d log|f'| on J_c vanishes -- equivalently, where the leading eigenvalue
    lambda(d) of the transfer operator (L_d g)(z) = sum_{f(w)=z} |f'(w)|^-d g(w)
    equals 1. lambda is approximated by power iteration realized on the exact
    backward-orbit tree of the repelling fixed point beta = (1 + sqrt(1-4c))/2
    (a point of J_c for every c): pulling beta back k levels through the two
    inverse branches +-sqrt(z - c) gives all 2**k preimages, and

        S_k(d) = sum over level-k preimages x of |(f^k)'(x)|^-d

    is exactly L_d^k applied to the constant function, evaluated at beta. The
    m-step ratio (S_depth / S_{depth-m})**(1/m) -- the geometric mean of m
    consecutive one-step ratios -- converges to lambda(d) at the transfer
    operator's spectral-gap rate; the averaging matters because subleading
    eigenvalues are often complex (e.g. the rabbit's period-3 combinatorics),
    which makes the single-step ratio oscillate with depth instead of
    settling. d is then found by solving lambda(d) = 1 with a clipped Newton
    iteration on log lambda (monotone decreasing in d, since its d-derivative
    is minus the Lyapunov exponent of the conformal measure, hence negative).

    Unlike box-counting a rendered image, this never touches pixels, so it
    has no resolution floor: the Cantor-dust Julia sets outside M, whose gap
    structure lies far below any feasible pixel size, get their true
    dimension (< 1 along the real slices, dropping toward 0 as c -> infinity)
    instead of an estimate contaminated by the nearby connected sets'. At
    non-hyperbolic parameters (parabolic c, Misiurewicz points, the boundary
    itself) Bowen's hypotheses fail and the value is only an approximation,
    but those parameters form a measure-zero set that a generic grid misses.

    |(f^k)'| is accumulated in log space along each branch: the chain rule
    gives log|(f^k)'(x)| = sum of log(2|z_j|) over x's ancestors below beta
    plus x itself. A branch passing within 1e-6 of the critical point 0
    (possible only when J_c nearly meets 0, i.e. c essentially on the
    boundary) would contribute log ~ -inf and an exploding weight; such
    branches are pruned by assigning a large sentinel log-derivative, which
    zeroes their weight and, being inherited additively, their descendants'.
    """
    b = len(c_batch)
    c = np.asarray(c_batch, dtype=complex)[:, None]
    z = 0.5 * (1 + np.sqrt(1 - 4 * c))          # beta, level-0 tree (b, 1)
    logD = np.zeros((b, 1))
    for level in range(depth):
        w = np.sqrt(z - c)
        z = np.concatenate([w, -w], axis=1)     # both inverse branches
        a = np.abs(z)
        step = np.where(a > 1e-6, np.log(2 * np.maximum(a, 1e-30)), 1e3)
        logD = np.concatenate([logD, logD], axis=1) + step
        if level == depth - m - 1:
            logD_prev = logD                    # level depth-m weights

    # Newton on g(d) = (log S_depth(d) - log S_{depth-m}(d)) / m = log lambda.
    # g' = -(<logD>_depth - <logD>_prev) / m under the respective Boltzmann
    # weights; clipping keeps iterates in the geometrically possible [0, 2].
    d = np.full(b, 1.0)
    for _ in range(30):
        wn = np.exp(-d[:, None] * logD)
        wp = np.exp(-d[:, None] * logD_prev)
        sn, sp = wn.sum(axis=1), wp.sum(axis=1)
        g = (np.log(sn) - np.log(sp)) / m
        gp = ((wp * logD_prev).sum(axis=1) / sp
              - (wn * logD).sum(axis=1) / sn) / m
        d = np.clip(d - g / np.minimum(gp, -1e-12), 0.0, 2.0)
        if np.max(np.abs(g)) < 1e-9:
            break
    return d


class MandelJulia:
    """Two linked panels:

      left  = Mandelbrot-type set { c : orbit of z0 under z -> z**2 + c is bounded },
              with a draggable marker at the Julia parameter c;
      right = Julia set { z0 : orbit of z0 under z -> z**2 + c is bounded },
              with a draggable marker at the Mandelbrot initial condition z0.

    Dragging the marker in one panel recomputes the *other* panel:
      - moving c (in the Mandelbrot panel) redraws the Julia set for that c;
      - moving z0 (in the Julia panel) redraws the Mandelbrot set seeded at that z0.
    """

    def __init__(self, c=-0.8 + 0.156j, z0=0 + 0j):
        self.c = c      # Julia parameter, chosen in the Mandelbrot panel
        self.z0 = z0    # Mandelbrot initial condition, chosen in the Julia panel

        # Current viewing windows (mutated by zooming); the module-level
        # constants above remain the "home" views that `r` restores.
        self.extent_m = MANDEL_EXTENT
        self.extent_j = JULIA_EXTENT

        # Display mode for the Mandelbrot panel:
        #   'escape'     - smooth escape-time shading of z0's orbit (default)
        #   'difference' - classify each c by which of the two orbits (z0's and
        #                  the critical point's) stays bounded, isolating
        #                  M ∖ M_z0 and hence the boundary of {c : z0 ∈ K_c}
        #   'level'      - the solid finite-time survival set S_T ("un-dustified"
        #                  M_z0); S_T shrinks onto M_z0 (dust included) as T grows
        #   'dimension'  - heat map of the Hausdorff dimension of the Julia
        #                  set J_c at each parameter c (independent of z0)
        self.mode_m = 'escape'
        self.level_T = LEVEL_T0   # iteration threshold for 'level' mode

        # Dimension-mode cache: (settings key, dimension array). The map is a
        # function of the viewing window alone -- not of z0 or c -- so marker
        # drags reuse it; only zooming (or the hi-res export) recomputes.
        self._dim_cache = None
        self.cbar = None          # colorbar shown only in dimension mode

        self.fig, (self.ax_m, self.ax_j) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle('Left-drag: move point  ·  Right-drag: zoom  ·  '
                          'r: reset view  ·  e: export hi-res PNG\n'
                          'd: difference mode  ·  u: survival mode (↑/↓ adjust T)'
                          '  ·  b: Julia-dimension map (slow to compute)')

        # Initial fine-quality renders.
        self.im_m = self.ax_m.imshow(self._mandel_image(FINE), origin='lower',
                                     extent=self.extent_m, cmap=CMAP)
        self.im_j = self.ax_j.imshow(self._julia_image(FINE), origin='lower',
                                     extent=self.extent_j, cmap=CMAP)

        # Draggable markers: c lives in the Mandelbrot panel, z0 in the Julia panel.
        # Cyan sits opposite inferno's black->red->yellow sweep, so the marker stays
        # visible over every part of the image; the black edge separates it from
        # bright-yellow boundary filaments.
        self.pt_c, = self.ax_m.plot([self.c.real], [self.c.imag], 'o',
                                    color='cyan', markersize=8, markeredgecolor='black')
        self.pt_z0, = self.ax_j.plot([self.z0.real], [self.z0.imag], 'o',
                                     color='cyan', markersize=8, markeredgecolor='black')

        self._update_titles()

        self.dragging = None          # None, 'c', or 'z0'
        self._last_draw = 0.0         # timestamp of the last coarse recomputation

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Rubber-band zoom rectangles, one per panel, bound to the RIGHT mouse
        # button so they can't collide with left-button marker drags.
        # minspan (in *pixels*, so it survives zooming) discards accidental
        # right-clicks that would otherwise select a degenerate window.
        # NOTE: RectangleSelector binds 'r' to "rotate selection" by default,
        # which collides with our reset key -- pressing r would silently put
        # both selectors into rotation mode and break all subsequent zooms.
        # Mapping rotate to '' (matches no key) disables it.
        selector_style = dict(edgecolor='cyan', facecolor='none',
                              linewidth=1.2, linestyle='--')
        no_rotate = dict(rotate='')
        self.sel_m = RectangleSelector(
            self.ax_m, lambda ec, er: self._on_zoom('mandel'),
            useblit=True, button=[3], minspanx=5, minspany=5,
            spancoords='pixels', props=selector_style,
            state_modifier_keys=no_rotate)
        self.sel_j = RectangleSelector(
            self.ax_j, lambda ec, er: self._on_zoom('julia'),
            useblit=True, button=[3], minspanx=5, minspany=5,
            spancoords='pixels', props=selector_style,
            state_modifier_keys=no_rotate)

    # ---- rendering ------------------------------------------------------

    @staticmethod
    def _effective_iters(quality, extent, home_extent):
        """Scale the iteration budget with magnification.

        Near the set's boundary, the number of iterations needed to decide
        escape grows as you zoom in; with a fixed budget, deep zooms degrade
        into featureless "everything looks interior" blobs. Structure scales
        with the *logarithm* of magnification, so a log-proportional boost
        keeps detail resolved without exploding compute on shallow zooms.
        """
        zoom = (home_extent[1] - home_extent[0]) / (extent[1] - extent[0])
        if zoom <= 1:
            return quality['max_iter']
        return int(quality['max_iter'] * (1 + 0.6 * np.log10(zoom)))

    def _mandel_image(self, quality):
        if self.mode_m == 'dimension':
            # The dimension map has its own (much coarser) resolution settings;
            # `quality` only selects between the interactive and export tiers.
            return self._dimension_image(
                DIM_HIRES if quality['n'] >= HIRES['n'] else DIM_QUALITY)
        C = complex_grid(self.extent_m, quality['n'])
        iters = self._effective_iters(quality, self.extent_m, MANDEL_EXTENT)
        if self.mode_m == 'level':
            # Survival mode: threshold at exactly T iterations (the zoom-scaled
            # budget is irrelevant here; T itself is the resolution dial).
            mu_T = escape_time(self.z0, C, self.level_T)
            mu_crit = escape_time(0, C, iters)
            return self._level_rgb(mu_T, mu_crit)
        mu_z0 = escape_time(self.z0, C, iters)
        if self.mode_m == 'escape':
            return shade(mu_z0)
        # Difference mode: also iterate the critical orbit (z0 = 0) over the
        # same c-grid, i.e. compute the classical Mandelbrot set alongside.
        mu_crit = escape_time(0, C, iters)
        return self._difference_rgb(mu_z0, mu_crit)

    @staticmethod
    def _level_rgb(mu_T, mu_crit):
        """Render the solid finite-time survival set S_T ("un-dustified" M_z0).

        S_T = {c : z0's orbit survives >= T iterations} is compact and solid,
        and the family is nested with intersection over all T equal to M_z0.
        Coloring:
          survives, c in M      -> black (converges to the visible M_z0 ∩ M)
          survives, c not in M  -> green (the flesh over the dust: melts away
                                   as T grows, leaving the Cantor skeleton)
          escaped before T      -> white
        """
        surv = mu_T == 0
        in_m = mu_crit == 0
        rgb = np.ones(surv.shape + (3,))          # white background
        rgb[surv & in_m] = COLOR_BOTH
        rgb[surv & ~in_m] = COLOR_Z0_ONLY
        return rgb

    @staticmethod
    def _difference_rgb(mu_z0, mu_crit):
        """Classify each parameter c by which orbits stayed bounded.

        mu == 0 marks a bounded (non-escaped) orbit. Four classes:
          both bounded        -> black  (M_z0 ∩ M: z0's stability survives in M)
          only critical (M∖M_z0) -> blue; its inner edge is the boundary of the
                                 z0-dependent set {c : z0 ∈ K_c} inside M
          only z0             -> green (the measure-zero "dust": z0 sits on a
                                 Cantor Julia set; should be virtually empty)
          neither             -> smooth escape shading of z0's orbit, as usual
        """
        base = shade(mu_z0)
        top = base.max()
        rgb = colormaps[CMAP](base / top if top > 0 else base)[..., :3]
        in_s = mu_z0 == 0     # z0's orbit bounded:      c ∈ S(z0) = {c : z0 ∈ K_c}
        in_m = mu_crit == 0   # critical orbit bounded:  c ∈ M
        rgb[in_m & in_s] = COLOR_BOTH
        rgb[in_m & ~in_s] = COLOR_CRIT_ONLY
        rgb[~in_m & in_s] = COLOR_Z0_ONLY
        return rgb

    def _julia_image(self, quality):
        Z0 = complex_grid(self.extent_j, quality['n'])
        iters = self._effective_iters(quality, self.extent_j, JULIA_EXTENT)
        mu = escape_time(Z0, self.c, iters)
        return shade(mu)

    def _dimension_image(self, settings):
        """Hausdorff dimension of J_c over the current parameter window.

        Returns a (p, p) float array: entry [i, j] is the Bowen-formula
        dimension (see pressure_dimension) of the Julia set for the parameter
        c at that grid cell. Computed in row batches sized so each holds at
        most DIM_BATCH preimage-tree leaves, and cached: the map depends only
        on the viewing window and the settings (not on z0 or c), so marker
        drags hit the cache and only zooming or the hi-res export triggers a
        recomputation. Progress is printed to the console.
        """
        key = (tuple(self.extent_m), settings['p'], settings['depth'])
        if self._dim_cache is not None and self._dim_cache[0] == key:
            return self._dim_cache[1]
        p, depth = settings['p'], settings['depth']
        C = complex_grid(self.extent_m, p)
        dims = np.empty((p, p))
        # dim J_c = dim J_conj(c): conjugating c mirrors the Julia set, which
        # leaves its dimension unchanged. In a window symmetric about the
        # real axis (like the home view) only the top half is computed and
        # the bottom half is mirrored -- an exact 2x saving.
        ymin, ymax = self.extent_m[2], self.extent_m[3]
        symmetric = abs(ymin + ymax) < 1e-9 * (ymax - ymin)
        start = p // 2 if symmetric else 0
        total = p - start
        chunk = max(1, DIM_BATCH // (p * 2 ** depth))
        print(f'computing {p}x{p} dimension map '
              f'({total} rows of {p} preimage trees, depth {depth})...')
        t0 = time.perf_counter()
        done = 0
        report = total / 4
        for r in range(start, p, chunk):
            rows = C[r:r + chunk]
            dims[r:r + chunk] = pressure_dimension(
                rows.ravel(), depth).reshape(rows.shape)
            done += rows.shape[0]
            if done >= report and done < total:
                print(f'  {done}/{total} rows, '
                      f'{time.perf_counter() - t0:.0f}s elapsed')
                report += total / 4
        if symmetric:
            dims[:start] = dims[::-1][:start].copy()   # rows k <- p-1-k
        print(f'  done in {time.perf_counter() - t0:.0f}s')
        self._dim_cache = (key, dims)
        return dims

    def _dim_at(self, c):
        """Dimension estimate at the grid parameter nearest c, from the cache."""
        if self._dim_cache is None:
            return None
        (x0, x1, y0, y1), p = self._dim_cache[0][0], self._dim_cache[0][1]
        img = self._dim_cache[1]
        i = int(round((c.imag - y0) / (y1 - y0) * (p - 1)))
        j = int(round((c.real - x0) / (x1 - x0) * (p - 1)))
        if 0 <= i < p and 0 <= j < p:
            return img[i, j]
        return None

    def _redraw_mandel(self, quality):
        img = self._mandel_image(quality)
        self.im_m.set_data(img)
        if img.ndim == 2:   # scalar field -> colormap; clim is meaningless for RGB
            self.im_m.set_clim(img.min(), img.max())

    def _redraw_julia(self, quality):
        img = self._julia_image(quality)
        self.im_j.set_data(img)
        self.im_j.set_clim(img.min(), img.max())

    def _update_titles(self):
        if self.mode_m == 'dimension':
            d = self._dim_at(self.c)
            at_marker = '' if d is None else f' · d ≈ {d:.2f} at marker'
            self.ax_m.set_title('Hausdorff dimension of $J_c$ (Bowen pressure '
                                'formula)' + at_marker, fontsize=10)
            self.ax_j.set_title(f'Julia set, parameter '
                                f'$c$ = {self.c.real:+.4f} {self.c.imag:+.4f}i')
            return
        if self.mode_m == 'level':
            self.ax_m.set_title(
                f'survival set $S_T$, T = {self.level_T} iterations · '
                f'green: $S_T \\setminus M$   ($z_0$ = {self.z0.real:+.4f} '
                f'{self.z0.imag:+.4f}i)', fontsize=10)
            return
        if self.mode_m == 'difference':
            self.ax_m.set_title(
                f'black: $M_{{z_0}} \\cap M$ · blue: $M \\setminus M_{{z_0}}$ · '
                f'green: dust   ($z_0$ = {self.z0.real:+.4f} {self.z0.imag:+.4f}i)',
                fontsize=10)
            return
        self.ax_m.set_title(f'Mandelbrot set, initial condition '
                            f'$z_0$ = {self.z0.real:+.4f} {self.z0.imag:+.4f}i')
        self.ax_j.set_title(f'Julia set, parameter '
                            f'$c$ = {self.c.real:+.4f} {self.c.imag:+.4f}i')

    def _apply_point(self, which, x, y, quality):
        """Move the marker `which` to (x, y) and recompute the linked panel."""
        if which == 'c':
            self.c = x + 1j * y
            self.pt_c.set_data([x], [y])
            self._redraw_julia(quality)
        else:
            self.z0 = x + 1j * y
            self.pt_z0.set_data([x], [y])
            self._redraw_mandel(quality)
        self._update_titles()
        self.fig.canvas.draw_idle()

    # ---- zooming --------------------------------------------------------

    def _set_view(self, which, extent):
        """Point `which` panel at a new complex-plane window and re-render it.

        This is a true zoom: the escape-time grid is recomputed over the new
        window at fine quality (with an iteration budget scaled to the
        magnification), not a stretch of already-rendered pixels.

        In dimension mode the export uses the DIM_HIRES settings instead, so
        the file is p x p (one pixel per parameter) -- much smaller than the
        escape-mode exports, and much slower to compute.
        """
        if which == 'mandel':
            self.extent_m = extent
            self._redraw_mandel(FINE)
            self.im_m.set_extent(extent)
            ax = self.ax_m
        else:
            self.extent_j = extent
            self._redraw_julia(FINE)
            self.im_j.set_extent(extent)
            ax = self.ax_j
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        self.fig.canvas.draw_idle()

    def _on_zoom(self, which):
        """RectangleSelector callback: zoom `which` panel to the drawn box."""
        sel = self.sel_m if which == 'mandel' else self.sel_j
        xmin, xmax, ymin, ymax = sel.extents   # corners, already sorted
        if xmax <= xmin or ymax <= ymin:       # degenerate selection
            return
        sel.set_visible(False)                 # drop the rubber-band outline
        self._set_view(which, (xmin, xmax, ymin, ymax))

    def save_panel(self, which, path=None, quality=HIRES):
        """Recompute `which` panel ('mandel' or 'julia') at export quality and
        save it as a PNG. Returns the path written.

        The image is recomputed from scratch over the panel's *current*
        viewing window (so it honors any zoom) with the current display mode,
        marker position, and (in survival mode) threshold T -- it is not an
        upscale of the on-screen 800x800 render. plt.imsave writes the array
        one grid point per pixel, so the file is exactly quality['n'] pixels
        square with no axes, titles, markers, or padding.
        """
        if which == 'mandel':
            img = self._mandel_image(quality)
        else:
            img = self._julia_image(quality)
        if path is None:
            path = f'{which}_{time.strftime("%Y%m%d-%H%M%S")}.png'
        if img.ndim == 2:
            # Scalar escape-time field: imsave normalizes min->max onto the
            # colormap, matching the set_clim scaling used on screen.
            plt.imsave(path, img, cmap=CMAP, origin='lower')
        else:
            # Difference/survival modes already produced an RGB array.
            plt.imsave(path, img, origin='lower')
        print(f'saved {which} panel ({img.shape[1]}x{img.shape[0]} px) -> {path}')
        return path

    def _switch_mode(self, mode):
        """Toggle the Mandelbrot panel between `mode` and plain escape view."""
        self.mode_m = mode if self.mode_m != mode else 'escape'
        self._redraw_mandel(FINE)
        # The dimension heat map is the only mode whose pixel values carry
        # quantitative meaning, so it gets a colorbar; remove it (restoring
        # the panel's original width) when leaving the mode.
        if self.mode_m == 'dimension' and self.cbar is None:
            self.cbar = self.fig.colorbar(self.im_m, ax=self.ax_m,
                                          fraction=0.046, pad=0.02)
        elif self.mode_m != 'dimension' and self.cbar is not None:
            self.cbar.remove()
            self.cbar = None
        self._update_titles()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """`r`: restore home window (hovered panel, else both).
        `d`: toggle difference mode.  `u`: toggle survival ("un-dustify") mode.
        up/down (in survival mode): raise/lower the iteration threshold T.
        `b`: toggle the Julia-dimension heat map.
        `e`: export a hi-res PNG of the hovered panel (else both panels)."""
        if event.key in ('b', 'B'):
            self._switch_mode('dimension')
            return
        if event.key in ('e', 'E'):
            if event.inaxes is self.ax_m:
                self.save_panel('mandel')
            elif event.inaxes is self.ax_j:
                self.save_panel('julia')
            else:
                self.save_panel('mandel')
                self.save_panel('julia')
            return
        if event.key in ('d', 'D'):
            self._switch_mode('difference')
            return
        if event.key in ('u', 'U'):
            self._switch_mode('level')
            return
        if event.key in ('up', 'down') and self.mode_m == 'level':
            if event.key == 'up':
                self.level_T = int(np.ceil(self.level_T * 1.5))
            else:
                self.level_T = max(5, int(self.level_T / 1.5))
            self._redraw_mandel(FINE)
            self._update_titles()
            self.fig.canvas.draw_idle()
            return
        if event.key not in ('r', 'R'):
            return
        if event.inaxes is self.ax_m:
            self._set_view('mandel', MANDEL_EXTENT)
        elif event.inaxes is self.ax_j:
            self._set_view('julia', JULIA_EXTENT)
        else:
            self._set_view('mandel', MANDEL_EXTENT)
            self._set_view('julia', JULIA_EXTENT)

    # ---- mouse events ---------------------------------------------------

    def on_press(self, event):
        if event.button != 1:
            return   # marker dragging is left-button only; right button zooms
        if event.inaxes is self.ax_m:
            self.dragging = 'c'
        elif event.inaxes is self.ax_j:
            self.dragging = 'z0'
        else:
            return
        # Clicking anywhere in a panel jumps its marker there and starts a drag.
        self._apply_point(self.dragging, event.xdata, event.ydata, COARSE)
        self._last_draw = time.perf_counter()

    def on_motion(self, event):
        if self.dragging is None or event.inaxes is None or event.xdata is None:
            return
        # Only respond to motion inside the panel being dragged.
        expected_ax = self.ax_m if self.dragging == 'c' else self.ax_j
        if event.inaxes is not expected_ax:
            return
        now = time.perf_counter()
        if now - self._last_draw < DRAG_THROTTLE:
            return
        self._last_draw = now
        self._apply_point(self.dragging, event.xdata, event.ydata, COARSE)

    def on_release(self, event):
        if self.dragging is None:
            return
        which = self.dragging
        self.dragging = None
        # Refine the linked panel at full quality using the final marker position.
        if which == 'c':
            self._redraw_julia(FINE)
        else:
            self._redraw_mandel(FINE)
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    viewer = MandelJulia()
    plt.show()
