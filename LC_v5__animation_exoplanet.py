import sys
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# ====================================================
# This is a companion to LC_v5.py: same physics (orbit_state, flux, Kepler
# solver), reproduced here so this file can run standalone. If you change
# star/orbit parameters in LC_v5.py, mirror those changes here too.
# ====================================================

# General Scientific Constants (DO NOT CHANGE)
AU = 1.5e11
G = 6.67e-11
M_Sol = 1.99e30
L_Sol = 3.9e26
R_Sol = 6.96e8
yr = 365 * 24 * 60 * 60

# INPUT PARAMETERS: Primary Star
target = "Simulated Transit of Earth across Sun"
m1 = 1
r1 = 1
L1 = 1
primary_color = 'orange'

# INPUT PARAMETERS: Secondary Star
m2 = 0.00000300150829563
r2 = 0.00916794
L2 = 0
secondary_color = 'blue'

# INPUT PARAMETERS: Orbital Elements
ORBIT_INPUT = "a"
P = 2.867328
a_AU = 1
i = 89.98
e = 0.00167
omega = 0.0

# --- Animation-specific settings ---
N_FRAMES = 1000          # frames per orbital period (higher = smoother motion, bigger file)
N_PERIODS = 1           # how many consecutive orbits to animate through
FPS = 30                # playback speed
N_SAMPLES_BG = 5 * 10**6   # resolution of the background light-curve trace, PER PERIOD (doesn't need to match LC_v5's 5e6)
# --------------------------------------------------
# Derived Quantities
if ORBIT_INPUT == "P":
    P = P * 24 * 60 * 60
    sma = ((P / yr)**2 * (m1 + m2))**(1 / 3) * AU
elif ORBIT_INPUT == "a":
    sma = a_AU * AU
    P = np.sqrt((sma / AU)**3 / (m1 + m2)) * yr
else:
    raise ValueError("ORBIT_INPUT must be 'P' or 'a'")

OUTPUT_FILE = f"{target}_{P/(24*60**2):.3f}d_{sma/AU:.3f}AU_{N_FRAMES}_{e}_{N_FRAMES}frames_{FPS}.mp4"

a_Rsol = sma / R_Sol
w = (2 * np.pi) / P
L_total = L1 + L2
A1 = np.pi * r1**2
A2 = np.pi * r2**2


def solve_kepler(M, e, tol=1e-12, max_iter=200):
    M = np.atleast_1d(np.asarray(M, dtype=float))
    E = M.copy() if e < 0.8 else np.full_like(M, np.pi)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = f / fp
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def true_anomaly(E, e):
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def orbit_state(t):
    M = np.mod(w * np.asarray(t, dtype=float), 2 * np.pi)
    E = solve_kepler(M, e)
    nu = true_anomaly(E, e)
    r = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu))
    om = np.radians(omega)
    inc = np.radians(i)
    x = r * np.cos(om + nu)
    y = r * np.sin(om + nu) * np.cos(inc)
    z = r * np.sin(om + nu) * np.sin(inc)
    d = np.sqrt(x**2 + y**2)
    return x, y, z, d, r, nu


def A_c(d):
    d = np.atleast_1d(np.asarray(d, dtype=float))
    area = np.zeros_like(d)
    outside = d >= (r1 + r2)
    inside = d <= abs(r1 - r2)
    overlap = (~outside) & (~inside)
    area[inside] = np.pi * min(r1, r2)**2
    if np.any(overlap):
        dv = d[overlap]
        arg1 = np.clip((dv**2 + r1**2 - r2**2) / (2 * dv * r1), -1.0, 1.0)
        arg2 = np.clip((dv**2 + r2**2 - r1**2) / (2 * dv * r2), -1.0, 1.0)
        term1 = r1**2 * np.arccos(arg1)
        term2 = r2**2 * np.arccos(arg2)
        term3 = 0.5 * np.sqrt(np.clip((dv**2 - (r2 - r1)**2) * ((r1 + r2)**2 - dv**2), 0.0, None))
        area[overlap] = term1 + term2 - term3
    return area


def flux(d, z):
    d = np.atleast_1d(np.asarray(d, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    L = np.full_like(d, L_total)
    eclipsing = d < (r1 + r2)
    if np.any(eclipsing):
        dc = d[eclipsing]
        zc = z[eclipsing]
        Ac = A_c(dc)
        Lc = np.empty_like(dc)
        total = dc <= abs(r1 - r2)
        front2 = zc > 0
        front1 = ~front2

        m = front2 & total
        Lc[m] = L2 if r2 >= r1 else L2 + (A1 - A2) / A1 * L1
        m = front2 & ~total
        Lc[m] = L2 + (A1 - Ac[m]) / A1 * L1

        m = front1 & total
        Lc[m] = L1 if r1 >= r2 else L1 + (A2 - A1) / A2 * L2
        m = front1 & ~total
        Lc[m] = L1 + (A2 - Ac[m]) / A2 * L2

        L[eclipsing] = Lc
    return L


# --------------------------------------------------
# Background light curve (fine sampling, drawn once as a static reference trace),
# spanning all N_PERIODS periods that will be animated.
t_bg = np.linspace(0, N_PERIODS * P, N_SAMPLES_BG * N_PERIODS)
x_bg, y_bg, z_bg, d_bg, r_bg, nu_bg = orbit_state(t_bg)
L_bg = flux(d_bg, z_bg)

# --------------------------------------------------
# Eclipse identification (single period), needed for the title -- same depth-based
# primary/secondary convention as LC_v5.py: "primary" = whichever eclipse is deeper.
N1 = N_SAMPLES_BG  # first period's worth of samples out of t_bg
t_p1, x_p1, y_p1, z_p1, d_p1, L_p1 = t_bg[:N1], x_bg[:N1], y_bg[:N1], z_bg[:N1], d_bg[:N1], L_bg[:N1]

eclipse_mask_p1 = d_p1 < (r1 + r2)
primary_mask_p1 = eclipse_mask_p1 & (z_p1 > 0)
secondary_mask_p1 = eclipse_mask_p1 & (z_p1 < 0)


def get_segments(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    return np.split(idx, splits + 1)


def eclipse_info(segs, t, d, L):
    """Duration, L_min, and d_min (used as an approximate impact parameter) for
    the largest contiguous eclipse segment -- same quantities LC_v5.py reports."""
    if not segs:
        return None
    seg = max(segs, key=len)
    mid_idx = seg[np.argmin(L[seg])]
    return {
        't_start': t[seg[0]], 't_end': t[seg[-1]],
        'L_min': L[mid_idx], 'd_min': d[mid_idx],
        'duration': t[seg[-1]] - t[seg[0]],
    }


info_zpos = eclipse_info(get_segments(primary_mask_p1), t_p1, d_p1, L_p1)
info_zneg = eclipse_info(get_segments(secondary_mask_p1), t_p1, d_p1, L_p1)

# Conjunction geometry (always available -- diagnostics still print with no eclipse)
nu_c_zpos = np.radians(90.0 - omega)
nu_c_zneg = np.radians(270.0 - omega)
r_c_zpos = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu_c_zpos))
r_c_zneg = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu_c_zneg))

# "Primary" = whichever eclipse is deeper (lower L_min), matching LC_v5.py's convention.
# Conjunction (nu_c, r_c) follow that same depth-based assignment; if no eclipse occurs,
# keep the geometric z>0 / z<0 labeling so impact parameter / i_min still print.
if info_zpos and info_zneg:
    if info_zpos['L_min'] <= info_zneg['L_min']:
        pe, se = info_zpos, info_zneg
        nu_c_pe, r_c_pe = nu_c_zpos, r_c_zpos
        nu_c_se, r_c_se = nu_c_zneg, r_c_zneg
    else:
        pe, se = info_zneg, info_zpos
        nu_c_pe, r_c_pe = nu_c_zneg, r_c_zneg
        nu_c_se, r_c_se = nu_c_zpos, r_c_zpos
elif info_zpos:
    pe, se = info_zpos, None
    nu_c_pe, r_c_pe = nu_c_zpos, r_c_zpos
    nu_c_se, r_c_se = nu_c_zneg, r_c_zneg
elif info_zneg:
    pe, se = info_zneg, None
    nu_c_pe, r_c_pe = nu_c_zneg, r_c_zneg
    nu_c_se, r_c_se = nu_c_zpos, r_c_zpos
else:
    pe, se = None, None
    nu_c_pe, r_c_pe = nu_c_zpos, r_c_zpos
    nu_c_se, r_c_se = nu_c_zneg, r_c_zneg

pe_Lmin = pe['L_min'] if pe else None
se_Lmin = se['L_min'] if se else None

# Animate starting shortly before the transit (planet events are a tiny fraction of P).
if pe is not None:
    t_mid = 0.5 * (pe['t_start'] + pe['t_end'])
    half = max(1.5 * pe['duration'], 0.002 * P)
    t_view0, t_view1 = t_mid - half, t_mid + half
    n_anim_frames = N_FRAMES
else:
    t_view0, t_view1 = 0.0, N_PERIODS * P
    n_anim_frames = N_FRAMES * N_PERIODS

t_frames = np.linspace(t_view0, t_view1, n_anim_frames, endpoint=False)
x_f, y_f, z_f, d_f, r_f, nu_f = orbit_state(t_frames)
L_f = flux(d_f, z_f)

# --------------------------------------------------
# Figure setup
fig, (ax_orbit, ax_lc) = plt.subplots(1, 2, figsize=(13, 6))

title_str = (
    f"{target}\n"
    rf"m₁ = {m1} $M_\odot$, r₁ = {r1} $R_\odot$, L₁ = {L1} $L_\odot$,    "
    rf"m₂ = {m2} $M_\odot$, r₂ = {r2} $R_\odot$, L₂ = {L2} $L_\odot$" + "\n"
    f"P = {P/(24*60**2):.4f} d, a = {sma/AU:.4f} AU, e = {e:.4f}, ω = {omega:.4f}°, i = {i:.4f}°\n"
)
if pe_Lmin is not None or se_Lmin is not None:
    parts = []
    if pe_Lmin is not None:
        parts.append(rf"Primary Eclipse L$_{{min}}$ = {pe_Lmin:.4f} $L_\odot$")
    if se_Lmin is not None:
        parts.append(rf"Secondary Eclipse L$_{{min}}$ = {se_Lmin:.4f} $L_\odot$")
    title_str += "    ".join(parts) + "\n"

    parts = []
    if pe or se:
        parts.append(rf"Eclipse Duration = {pe['duration']/60:.4f} min, b = {pe['d_min']/r1:.4f}")
    title_str += "    ".join(parts)
else:
    title_str += "No Eclipse Occurs"

fig.suptitle(title_str)
fig.subplots_adjust(top=0.75, wspace=0.3)

# Orbit panel
ax_orbit.set_xlim(-1.2*r1, 1.2*r1)
ax_orbit.set_ylim(-1.2*r1, 1.2*r1)
ax_orbit.set_aspect('equal')
ax_orbit.set_xlabel(r"Solar Radii $R_\odot$")
ax_orbit.set_ylabel(r"Solar Radii $R_\odot$")
ax_orbit.grid(True, alpha=0.3)

nu_full = np.linspace(0, 2 * np.pi, 500)
r_full = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu_full))
om = np.radians(omega)
inc = np.radians(i)
orbit_x = r_full * np.cos(om + nu_full)
orbit_y = r_full * np.sin(om + nu_full) * np.cos(inc)
ax_orbit.plot(orbit_x, orbit_y, color='black', lw=1, zorder=1)

star1_patch = Circle((0, 0), r1, color=primary_color, zorder=2)
star2_patch = Circle((x_f[0], y_f[0]), r2, color=secondary_color, zorder=3)
ax_orbit.add_patch(star1_patch)
ax_orbit.add_patch(star2_patch)
ax_orbit.legend(
    handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=primary_color, markersize=10, label='m1'),
             Line2D([0], [0], marker='o', color='w', markerfacecolor=secondary_color, markersize=10, label='m2')],
    loc='upper right'
)
time_text = ax_orbit.text(0.02, 0.02, '', transform=ax_orbit.transAxes)

# Light curve panel -- same time window as the animation; y scaled to the dip
ax_lc.plot(t_bg / P, L_bg, color='black', lw=1)
L_min_plot = float(np.min(L_bg))
depth = max(L_total - L_min_plot, 1e-6 * L_total)
y_pad = max(0.35 * depth, 1e-4 * L_total)
ax_lc.set_ylim(L_min_plot - y_pad, L_total + y_pad)
ax_lc.set_xlim(t_view0 / P, t_view1 / P)
ax_lc.set_xlabel("Phase")
ax_lc.set_ylabel("Solar Luminosities")
ax_lc.grid(True, alpha=0.3)

marker, = ax_lc.plot([], [], 'o', color='red', ms=8, zorder=3)
vline = ax_lc.axvline(0, color='gray', lw=1, ls='--')


def update(frame):
    x_t, y_t, z_t, L_t = x_f[frame], y_f[frame], z_f[frame], L_f[frame]
    phase = t_frames[frame] / P

    star2_patch.center = (x_t, y_t)
    # whichever star is nearer the observer (z >= 0 convention from LC_v5) draws on top
    star1_patch.set_zorder(2 if z_t >= 0 else 4)
    star2_patch.set_zorder(4 if z_t >= 0 else 2)

    marker.set_data([phase], [L_t])
    vline.set_xdata([phase, phase])
    time_text.set_text(f"phase = {phase:.3f}\nL = {L_t} $L_\\odot$")

    return star1_patch, star2_patch, marker, vline, time_text


ani = animation.FuncAnimation(fig, update, frames=n_anim_frames, interval=1000 / FPS, blit=False)

if OUTPUT_FILE.endswith('.gif'):
    ani.save(OUTPUT_FILE, writer='pillow', fps=FPS)
else:
    ani.save(OUTPUT_FILE, writer='ffmpeg', fps=FPS)

# --------------------------------------------------
# Diagnostic prints (same set as LC_v5.py; computed even when no eclipse occurs)
# Redirect to a logfile (same print formatting as before)
LOGFILE = f'logfile_{target}_{(P/86400):.3f}d_{sma/AU:.3f}AU_{e:.3f}.txt'
_log_file = open(LOGFILE, 'w', encoding='utf-8')
_stdout = sys.stdout
sys.stdout = _log_file

r_peri = a_Rsol * (1 - e)
print(f"Orbital Period: {P/(24*60*60):.4f} days")
print(f"Semi-major axis: {sma/AU:.4f} AU   (periastron: {r_peri:.4f} R☉, apastron: {a_Rsol*(1+e):.4f} R☉)")
print(f"Eccentricity: {e:.4f}   Argument of Periastron: {omega:.4f}°")
if pe:
    print(f"Primary Eclipse:   duration {pe['duration']/60:.4f} min, min separation {pe['d_min']:.4f} R☉, L_min = {pe['L_min']:.4f} L☉")
if se:
    print(f"Secondary Eclipse: duration {se['duration']/60:.4f} min, min separation {se['d_min']:.4f} R☉, L_min = {se['L_min']:.4f} L☉")
if pe is None and se is None:
    print("No eclipses occur for this geometry.")

inc_rad = np.radians(i)
Rsum = r1 + r2
Rdiff = r1 - r2


def eclipse_geometry(r_c, nu_c):
    """Analytic impact parameter, transit duration, and inclination thresholds for a
    single conjunction -- mirrors LC_v5.py so diagnostics still print with no eclipse."""
    b_c = r_c * np.cos(inc_rad)
    sin_i = np.sin(inc_rad)
    denom_c = 1 + e * np.cos(nu_c)

    if sin_i > 0:
        arg = np.clip(np.sqrt(max(Rsum**2 - b_c**2, 0.0)) / (r_c * sin_i), -1.0, 1.0)
        half_angle = np.arcsin(arg)
    else:
        half_angle = 0.0
    duration = half_angle * P * (1 - e**2)**1.5 / (np.pi * denom_c**2)

    i_min = np.degrees(np.arccos(np.clip(Rsum / r_c, -1.0, 1.0))) if r_c > 0 else np.nan
    i_grazing = np.degrees(np.arccos(np.clip(Rdiff / r_c, -1.0, 1.0))) if r_c > 0 else np.nan
    return {'b': b_c, 'duration': duration, 'i_min': i_min, 'i_grazing': i_grazing}


geo_pe = eclipse_geometry(r_c_pe, nu_c_pe)
geo_se = eclipse_geometry(r_c_se, nu_c_se)

a_min_Rsol = Rsum / (1 - e)
a_min_AU = a_min_Rsol * R_Sol / AU
P_min = np.sqrt((a_min_AU)**3 / (m1 + m2)) * yr

# Max sma for an eclipse at this i: r_c < (r1+r2)/|cos i|, then convert to a
# at each conjunction. Take the larger limit (at least one eclipse possible).
cos_i_abs = abs(np.cos(inc_rad))
if cos_i_abs > 1e-15:
    r_c_max = Rsum / cos_i_abs
    a_max_pe = r_c_max * (1 + e * np.cos(nu_c_pe)) / (1 - e**2)
    a_max_se = r_c_max * (1 + e * np.cos(nu_c_se)) / (1 - e**2)
    a_max_AU = max(a_max_pe, a_max_se) * R_Sol / AU
    P_max = np.sqrt((a_max_AU)**3 / (m1 + m2)) * yr
else:
    a_max_AU = np.inf
    P_max = np.inf

print(f"Primary Transit Duration: {geo_pe['duration']/60:.4f} minutes")
print(f"Primary Impact Parameter: {geo_pe['b']:.4f} R☉,    b/r₁ = {geo_pe['b']/r1:.4f}")
print(f"Primary Minimum Inclination for Eclipse: [{geo_pe['i_min']:.4f}°, {180-geo_pe['i_min']:.4f}°]")
print(f"Primary Minimum Grazing Eclipse Inclination: [{geo_pe['i_grazing']:.4f}°, {180-geo_pe['i_grazing']:.4f}°]")
print(f"Secondary Transit Duration: {geo_se['duration']/60:.4f} minutes")
print(f"Secondary Impact Parameter: {geo_se['b']:.4f} R☉,    b/r₁ = {geo_se['b']/r1:.4f}")
print(f"Secondary Minimum Inclination for Eclipse: [{geo_se['i_min']:.4f}°, {180-geo_se['i_min']:.4f}°]")
print(f"Secondary Minimum Grazing Eclipse Inclination: [{geo_se['i_grazing']:.4f}°, {180-geo_se['i_grazing']:.4f}°]")
if np.isfinite(a_max_AU):
    print(f"Possible Semi-Major Axis for Eclipse: {a_min_AU:.4f} <= a < {a_max_AU:.4f} AU")
    print(f"Minimum Possible Orbital Period: {P_min/(24*60*60):.4f} <= P < {P_max/(24*60*60):.4f} days")
else:
    print(f"Possible Semi-Major Axis for Eclipse: {a_min_AU:.4f} <= a < ∞ AU  (i ≈ 90°)")
    print(f"Minimum Possible Orbital Period: {P_min/(24*60*60):.4f} <= P < ∞ days  (i ≈ 90°)")

print(f"Animation saved to '{OUTPUT_FILE}'")

sys.stdout = _stdout
_log_file.close()
print(f"Log saved as '{LOGFILE}'")
