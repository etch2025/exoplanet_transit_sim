"""
Streamlit app: exoplanet / planet transit simulator
(based on LC_v5__animation_exoplanet.py).

Run:
    source venv/bin/activate
    streamlit run app.py
"""

import os
import shutil
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import streamlit as st


import hashlib
import pickle
from collections import OrderedDict

MAX_RENDER_CACHE_ENTRIES = 4  # GIF/MP4 bytes are multi-MB each; keep this small


def _cache_key(namespace, kwargs):
    """Stable key for manual caching (order-independent, no Streamlit involved)."""
    payload = pickle.dumps((namespace, sorted(kwargs.items())))
    return hashlib.md5(payload).hexdigest()


def cached_render(render_fn, namespace, kwargs, progress_callback=None):
    """Session-scoped, size-bounded LRU cache for render_gif/render_mp4.

    Deliberately NOT st.cache_data: that decorator records every Streamlit
    call made during a cache miss and replays them on a cache hit, which
    breaks when the recorded call references a progress-bar placeholder
    from a previous, now-stale script run (CacheReplayClosureError). This
    keeps the same "don't re-render identical params" benefit using a
    plain OrderedDict in session_state instead — capped so repeated
    slider tweaks within a session can't accumulate unbounded MB-sized
    GIF/MP4 blobs in memory.
    """
    store = st.session_state.setdefault("_render_cache", OrderedDict())
    key = _cache_key(namespace, kwargs)
    if key in store:
        store.move_to_end(key)  # mark as recently used
        return store[key]
    data = render_fn(**kwargs, progress_callback=progress_callback)
    store[key] = data
    store.move_to_end(key)
    while len(store) > MAX_RENDER_CACHE_ENTRIES:
        store.popitem(last=False)  # evict least-recently-used
    return data


def find_ffmpeg():
    """Locate ffmpeg: system install first, then the imageio-ffmpeg
    bundled static binary (works on Streamlit Cloud / any Linux host
    with no apt access)."""
    candidates = [
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    except Exception:
        pass

    return None

# ====================================================
# Constants
# ====================================================
AU = 1.5e11
R_Sol = 6.96e8
yr = 365 * 24 * 60 * 60
M_JUP_MSUN = 9.547919e-4   # M_Jup / M_Sun
R_JUP_RSUN = 0.102792       # R_Jup / R_Sun

st.set_page_config(page_title="Exoplanet Transit Simulator", layout="wide")


st.title("Exoplanet Transit Simulator Simulator")
st.caption("Powered by Matplotlib and NumPy.")
st.markdown("[Originally from the Eclipsing Binary Stars Simulator](https://github.com/etch2025/eclipsing_binary_stars_sim)")
st.subheader("Created by [Ethan Chen](https://www.chenastronomy.com/)", divider="gray")


# ====================================================
# Physics helpers (parameterized — no globals)
# ====================================================
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


def orbit_state(t, w, a_Rsol, e, omega, i):
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


def A_c(d, r1, r2):
    d = np.atleast_1d(np.asarray(d, dtype=float))
    area = np.zeros_like(d)
    outside = d >= (r1 + r2)
    inside = d <= abs(r1 - r2)
    overlap = (~outside) & (~inside)
    area[inside] = np.pi * min(r1, r2) ** 2
    if np.any(overlap):
        dv = d[overlap]
        arg1 = np.clip((dv**2 + r1**2 - r2**2) / (2 * dv * r1), -1.0, 1.0)
        arg2 = np.clip((dv**2 + r2**2 - r1**2) / (2 * dv * r2), -1.0, 1.0)
        term1 = r1**2 * np.arccos(arg1)
        term2 = r2**2 * np.arccos(arg2)
        term3 = 0.5 * np.sqrt(
            np.clip((dv**2 - (r2 - r1) ** 2) * ((r1 + r2) ** 2 - dv**2), 0.0, None)
        )
        area[overlap] = term1 + term2 - term3
    return area


def flux(d, z, r1, r2, L1, L2):
    L_total = L1 + L2
    A1 = np.pi * r1**2
    A2 = np.pi * r2**2
    d = np.atleast_1d(np.asarray(d, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    L = np.full_like(d, L_total)
    eclipsing = d < (r1 + r2)
    if np.any(eclipsing):
        dc = d[eclipsing]
        zc = z[eclipsing]
        Ac = A_c(dc, r1, r2)
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


def get_segments(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    return np.split(idx, splits + 1)


def eclipse_info(segs, t, d, L):
    if not segs:
        return None
    seg = max(segs, key=len)
    mid_idx = seg[np.argmin(L[seg])]
    return {
        "t_start": t[seg[0]],
        "t_end": t[seg[-1]],
        "t_mid": t[mid_idx],
        "L_min": L[mid_idx],
        "d_min": d[mid_idx],
        "duration": t[seg[-1]] - t[seg[0]],
    }


def time_of_true_anomaly(nu, w, e):
    """Orbital time (from periapsis) for a given true anomaly."""
    cos_E = (e + np.cos(nu)) / (1 + e * np.cos(nu))
    sin_E = (np.sqrt(max(1 - e**2, 0.0)) * np.sin(nu)) / (1 + e * np.cos(nu))
    E = np.arctan2(sin_E, cos_E)
    M = E - e * np.sin(E)
    return np.mod(M, 2 * np.pi) / w


def eclipse_geometry(r_c, nu_c, i, e, P, r1, r2):
    inc_rad = np.radians(i)
    Rsum = r1 + r2
    Rdiff = r1 - r2
    b_c = r_c * np.cos(inc_rad)
    sin_i = np.sin(inc_rad)
    denom_c = 1 + e * np.cos(nu_c)
    if sin_i > 0:
        arg = np.clip(np.sqrt(max(Rsum**2 - b_c**2, 0.0)) / (r_c * sin_i), -1.0, 1.0)
        half_angle = np.arcsin(arg)
    else:
        half_angle = 0.0
    duration = half_angle * P * (1 - e**2) ** 1.5 / (np.pi * denom_c**2)
    i_min = np.degrees(np.arccos(np.clip(Rsum / r_c, -1.0, 1.0))) if r_c > 0 else np.nan
    i_grazing = np.degrees(np.arccos(np.clip(Rdiff / r_c, -1.0, 1.0))) if r_c > 0 else np.nan
    return {"b": b_c, "duration": duration, "i_min": i_min, "i_grazing": i_grazing}


@st.cache_data(show_spinner="Running transit simulation…", max_entries=8, ttl=1800)
def run_simulation(
    m1, r1, L1, m2, r2, orbit_input, P_days, a_AU, i, e, omega,
    n_samples, n_frames, n_periods,
):
    L2 = 0.0

    if orbit_input == "P":
        P = P_days * 24 * 60 * 60
        sma = ((P / yr) ** 2 * (m1 + m2)) ** (1 / 3) * AU
    else:
        sma = a_AU * AU
        P = np.sqrt((sma / AU) ** 3 / (m1 + m2)) * yr

    a_Rsol = sma / R_Sol
    w = (2 * np.pi) / P
    L_total = L1 + L2

    t_bg = np.linspace(0, n_periods * P, n_samples * n_periods)
    x_bg, y_bg, z_bg, d_bg, r_bg, nu_bg = orbit_state(t_bg, w, a_Rsol, e, omega, i)
    L_bg = flux(d_bg, z_bg, r1, r2, L1, L2)

    N1 = n_samples
    t_p1, d_p1, z_p1, L_p1 = t_bg[:N1], d_bg[:N1], z_bg[:N1], L_bg[:N1]
    # Transit only: planet in front of the star (z > 0).
    transit_mask = (d_p1 < (r1 + r2)) & (z_p1 > 0)
    transit = eclipse_info(get_segments(transit_mask), t_p1, d_p1, L_p1)

    nu_c = np.radians(90.0 - omega)
    r_c = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu_c))
    t_conj = time_of_true_anomaly(nu_c, w, e)

    if transit is not None:
        t_mid = transit["t_mid"]
        half = max(1.5 * transit["duration"], 0.002 * P)
        t_view0, t_view1 = t_mid - half, t_mid + half
    else:
        t_mid = t_conj
        t_view0, t_view1 = 0.0, n_periods * P

    t_frames = np.linspace(t_view0, t_view1, n_frames, endpoint=False)
    x_f, y_f, z_f, d_f, r_f, nu_f = orbit_state(t_frames, w, a_Rsol, e, omega, i)
    L_f = flux(d_f, z_f, r1, r2, L1, L2)

    geo = eclipse_geometry(r_c, nu_c, i, e, P, r1, r2)

    Rsum = r1 + r2
    a_min_AU = (Rsum / (1 - e)) * R_Sol / AU
    P_min = np.sqrt(a_min_AU**3 / (m1 + m2)) * yr
    cos_i_abs = abs(np.cos(np.radians(i)))
    if cos_i_abs > 1e-15:
        r_c_max = Rsum / cos_i_abs
        a_max_AU = r_c_max * (1 + e * np.cos(nu_c)) / (1 - e**2) * R_Sol / AU
        P_max = np.sqrt(a_max_AU**3 / (m1 + m2)) * yr
    else:
        a_max_AU = np.inf
        P_max = np.inf

    return {
        "P": P,
        "sma": sma,
        "a_Rsol": a_Rsol,
        "L_total": L_total,
        "t_bg": t_bg,
        "L_bg": L_bg,
        "t_frames": t_frames,
        "x_f": x_f,
        "y_f": y_f,
        "z_f": z_f,
        "L_f": L_f,
        "t_view0": t_view0,
        "t_view1": t_view1,
        "t_mid": t_mid,
        "transit": transit,
        "geo": geo,
        "a_min_AU": a_min_AU,
        "a_max_AU": a_max_AU,
        "P_min": P_min,
        "P_max": P_max,
        "r_peri": a_Rsol * (1 - e),
        "r_ap": a_Rsol * (1 + e),
    }


def _animation_title(sim, target, m1, r1, L1, m2_jup, r2_jup, e, omega, i):
    P = sim["P"]
    sma = sim["sma"]
    transit = sim["transit"]
    title_str = (
        f"{target}\n"
        rf"m₁ = {m1:.6f} $M_\odot$, r₁ = {r1:.6f} $R_\odot$, L₁ = {L1:.6f} $L_\odot$,    "
        rf"m₂ = {m2_jup:.6f} $M_{{\rm J}}$, r₂ = {r2_jup:.6f} $R_{{\rm J}}$" + "\n"
        f"P = {P/(24*60**2):.6f} d, a = {sma/AU:.6f} AU, e = {e:.6f}, "
        f"ω = {omega:.6f}°, i = {i:.6f}°\n"
    )
    if transit is not None:
        title_str += (
            rf"Transit L$_{{min}}$ = {transit['L_min']:.6f} $L_\odot$" + "\n"
            rf"Transit Duration = {transit['duration']/60:.6f} min, "
            rf"b = {transit['d_min']/r1:.6f}"
        )
    else:
        title_str += "No Transit Occurs"
    return title_str


def _build_animation(
    sim, r1, r2, e, omega, i, primary_color, planet_color, target,
    m1, L1, m2_jup, r2_jup, fps,
):
    """Build a FuncAnimation for the transit window (shared by GIF and MP4)."""
    n_frames = len(sim["t_frames"])
    P = sim["P"]
    a_Rsol = sim["a_Rsol"]
    L_total = sim["L_total"]
    t_mid = sim["t_mid"]

    fig, (ax_orbit, ax_lc) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(_animation_title(sim, target, m1, r1, L1, m2_jup, r2_jup, e, omega, i))
    fig.subplots_adjust(top=0.78, wspace=0.3)

    ax_orbit.set_xlim(-1.2 * r1, 1.2 * r1)
    ax_orbit.set_ylim(-1.2 * r1, 1.2 * r1)
    ax_orbit.set_aspect("equal")
    ax_orbit.set_xlabel(r"Solar Radii $R_\odot$")
    ax_orbit.set_ylabel(r"Solar Radii $R_\odot$")
    ax_orbit.grid(True, alpha=0.3)

    nu_full = np.linspace(0, 2 * np.pi, 500)
    r_full = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu_full))
    om = np.radians(omega)
    inc = np.radians(i)
    ax_orbit.plot(
        r_full * np.cos(om + nu_full),
        r_full * np.sin(om + nu_full) * np.cos(inc),
        color="black", lw=1, zorder=1,
    )
    star1_patch = Circle((0, 0), r1, color=primary_color, zorder=2)
    star2_patch = Circle((sim["x_f"][0], sim["y_f"][0]), r2, color=planet_color, zorder=3)
    ax_orbit.add_patch(star1_patch)
    ax_orbit.add_patch(star2_patch)
    ax_orbit.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=primary_color, markersize=10, label="m1"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=planet_color, markersize=10, label="m2"),
        ],
        loc="upper right",
    )
    time_text = ax_orbit.text(0.02, 0.02, "", transform=ax_orbit.transAxes)

    phase_bg = (sim["t_bg"] - t_mid) / P
    ax_lc.plot(phase_bg, sim["L_bg"], color="black", lw=1)
    L_min_plot = float(np.min(sim["L_bg"]))
    depth = max(L_total - L_min_plot, 1e-6 * L_total)
    y_pad = max(0.35 * depth, 1e-4 * L_total)
    ax_lc.set_ylim(L_min_plot - y_pad, L_total + y_pad)
    ax_lc.set_xlim((sim["t_view0"] - t_mid) / P, (sim["t_view1"] - t_mid) / P)
    ax_lc.set_xlabel("Phase")
    ax_lc.set_ylabel("Solar Luminosities")
    ax_lc.grid(True, alpha=0.3)
    marker, = ax_lc.plot([], [], "o", color="red", ms=8, zorder=3)
    vline = ax_lc.axvline(0, color="gray", lw=1, ls="--")

    def update(frame):
        x_t = sim["x_f"][frame]
        y_t = sim["y_f"][frame]
        z_t = sim["z_f"][frame]
        L_t = sim["L_f"][frame]
        phase = (sim["t_frames"][frame] - t_mid) / P
        star2_patch.center = (x_t, y_t)
        star1_patch.set_zorder(2 if z_t >= 0 else 4)
        star2_patch.set_zorder(4 if z_t >= 0 else 2)
        marker.set_data([phase], [L_t])
        vline.set_xdata([phase, phase])
        time_text.set_text(f"phase = {phase:.6f}\nL = {L_t:.6f} $L_\\odot$")
        return star1_patch, star2_patch, marker, vline, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )
    return fig, ani


def render_gif(
    m1, r1, L1, m2_jup, r2_jup, orbit_input, P_days, a_AU, i, e, omega,
    n_samples, n_frames, primary_color, planet_color, target, fps,
    progress_callback=None,
):
    """Render GIF bytes (uncached — see `cached_render` for caching + progress).

    progress_callback(current_frame, total_frames) is called by
    matplotlib during encoding.
    """
    m2 = m2_jup * M_JUP_MSUN
    r2 = r2_jup * R_JUP_RSUN
    sim = run_simulation(
        m1, r1, L1, m2, r2, orbit_input, P_days, a_AU, i, e, omega,
        n_samples, n_frames, 1,
    )
    fig, ani = _build_animation(
        sim, r1, r2, e, omega, i, primary_color, planet_color, target,
        m1, L1, m2_jup, r2_jup, fps,
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        ani.save(
            tmp_path, writer="pillow", fps=fps, dpi=100,
            progress_callback=progress_callback,
        )
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        plt.close(fig)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return data


def render_mp4(
    m1, r1, L1, m2_jup, r2_jup, orbit_input, P_days, a_AU, i, e, omega,
    n_samples, n_frames, primary_color, planet_color, target, fps,
    progress_callback=None,
):
    """Render MP4 bytes (uncached — see `cached_render` for caching + progress).
    Requires ffmpeg.

    progress_callback(current_frame, total_frames) is called by
    matplotlib during encoding.
    """
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg not found. Install it (e.g. `brew install ffmpeg`) "
            "or add it to PATH."
        )
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path

    m2 = m2_jup * M_JUP_MSUN
    r2 = r2_jup * R_JUP_RSUN
    sim = run_simulation(
        m1, r1, L1, m2, r2, orbit_input, P_days, a_AU, i, e, omega,
        n_samples, n_frames, 1,
    )
    fig, ani = _build_animation(
        sim, r1, r2, e, omega, i, primary_color, planet_color, target,
        m1, L1, m2_jup, r2_jup, fps,
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        ani.save(
            tmp_path, writer=animation.FFMpegWriter(fps=fps), dpi=120,
            progress_callback=progress_callback,
        )
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        plt.close(fig)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    if not data:
        raise RuntimeError("ffmpeg produced an empty MP4.")
    return data


# ====================================================
# Sidebar controls
# ====================================================
with st.sidebar:
    st.header("System")
    target = st.text_input("Target name", "Simulated Jupiter Transit of Sun")

    st.subheader("Star (m₁)")
    m1 = st.number_input("Mass [M☉]", value=1.0, min_value=0.01, format="%.6f")
    r1 = st.number_input("Radius [R☉]", value=1.0, min_value=0.01, format="%.6f")
    L1 = st.number_input("Luminosity [L☉]", value=1.0, min_value=0.0, format="%.6f")
    primary_color = st.color_picker("Star color", "#FFA500")

    st.subheader("Planet (m₂)")
    m2_jup = st.number_input("Mass [M♃]", value=1.0, min_value=0.0, format="%.6f")
    r2_jup = st.number_input("Radius [R♃]", value=1.0, min_value=1e-6, format="%.6f")
    st.caption("Planet luminosity is fixed at 0. Mass/radius in Jupiter units.")
    planet_color = st.color_picker("Planet color", "#0000FF")

    st.subheader("Orbit")
    orbit_input = st.radio("Specify orbit by", ["a", "P"], horizontal=True, index=0)
    a_AU = st.number_input("Semi-major axis [AU]", value=5.2038, min_value=0.01, format="%.6f")
    P_days = st.number_input("Period [days]", value=4330.7845, min_value=0.01, format="%.6f")
    i = st.number_input("Inclination [deg]", 0.0, 180.0, 89.98, 0.01, format="%.6f")
    e = st.number_input("Eccentricity", 0.0, 0.9999, 0.0489, 0.00001, format="%.6f")
    omega = st.number_input("Argument of periastron [deg]", value=0.0, format="%.6f")

    st.subheader("Resolution")
    n_samples = st.number_input(
        "Light-curve samples / period", value=500_000, min_value=1_000, step=100_000,
    )
    n_frames = st.number_input(
        "Animation frames (transit window)", value=200, min_value=2, step=10,
    )
    fps = st.number_input("Animation FPS", value=30, min_value=1, step=1)

# Convert planet parameters from Jupiter → solar units for the physics engine.
m2 = m2_jup * M_JUP_MSUN
r2 = r2_jup * R_JUP_RSUN

# ====================================================
# Run simulation + GIF preview
# ====================================================
sim = run_simulation(
    m1, r1, L1, m2, r2, orbit_input, P_days, a_AU, i, e, omega,
    n_samples, n_frames, 1,
)

anim_kwargs = dict(
    m1=m1, r1=r1, L1=L1, m2_jup=m2_jup, r2_jup=r2_jup,
    orbit_input=orbit_input, P_days=P_days, a_AU=a_AU,
    i=i, e=e, omega=omega,
    n_samples=n_samples, n_frames=n_frames,
    primary_color=primary_color, planet_color=planet_color,
    target=target, fps=fps,
)

gif_progress_bar = st.progress(0.0, text="Rendering GIF frames…")


def _gif_progress(current_frame, total_frames):
    frac = (current_frame + 1) / total_frames
    gif_progress_bar.progress(
        frac, text=f"Rendering GIF frame {current_frame + 1}/{total_frames}"
    )


gif_bytes = cached_render(render_gif, "gif", anim_kwargs, progress_callback=_gif_progress)
gif_progress_bar.empty()  # cache hits skip the callback, so just clear it when done
st.image(gif_bytes, caption="Transit animation (phase 0 = mid-transit)", use_container_width=True)

base_name = (
    f"{target}_{sim['P']/(24*60**2):.6f}d_{sim['sma']/AU:.6f}AU_"
    f"{n_frames}_{e:.6f}_{fps}fps"
)

# ====================================================
# Diagnostics
# ====================================================
with st.expander("Diagnostics", expanded=True):
    P = sim["P"]
    st.write(f"**Orbital Period:** {P/(24*60*60):.6f} days")
    st.write(
        f"**Semi-major axis:** {sim['sma']/AU:.6f} AU   "
        f"(periastron: {sim['r_peri']:.6f} R☉, apastron: {sim['r_ap']:.6f} R☉)"
    )
    st.write(f"**Eccentricity:** {e:.6f}   **Argument of Periastron:** {omega:.6f}°")

    transit = sim["transit"]
    if transit:
        st.write(
            f"**Transit:** duration {transit['duration']/60:.6f} min, "
            f"min separation {transit['d_min']:.6f} R☉, L_min = {transit['L_min']:.6f} L☉"
        )
    else:
        st.write("No transit occurs for this geometry.")

    geo = sim["geo"]
    st.write(f"**Transit Duration:** {geo['duration']/60:.6f} minutes")
    st.write(
        f"**Impact Parameter:** {geo['b']:.6f} R☉,    "
        f"b/r₁ = {geo['b']/r1:.6f}"
    )
    st.write(
        f"**Minimum Inclination for Transit:** "
        f"[{geo['i_min']:.6f}°, {180-geo['i_min']:.6f}°]"
    )
    st.write(
        f"**Minimum Grazing Transit Inclination:** "
        f"[{geo['i_grazing']:.6f}°, {180-geo['i_grazing']:.6f}°]"
    )
    if np.isfinite(sim["a_max_AU"]):
        st.write(
            f"**Possible Semi-Major Axis for Transit:** "
            f"{sim['a_min_AU']:.6f} <= a < {sim['a_max_AU']:.6f} AU"
        )
        st.write(
            f"**Possible Orbital Period for Transit:** "
            f"{sim['P_min']/(24*60*60):.6f} <= P < {sim['P_max']/(24*60*60):.6f} days"
        )
    else:
        st.write(
            f"**Possible Semi-Major Axis for Transit:** "
            f"{sim['a_min_AU']:.6f} <= a < ∞ AU  (i ≈ 90°)"
        )
        st.write(
            f"**Possible Orbital Period for Transit:** "
            f"{sim['P_min']/(24*60*60):.6f} <= P < ∞ days  (i ≈ 90°)"
        )

# ====================================================
# Export GIF / MP4
# ====================================================
st.subheader("Export")
col_gif, col_mp4 = st.columns(2)

with col_gif:
    st.download_button(
        "Download GIF",
        data=gif_bytes,
        file_name=f"{base_name}.gif",
        mime="image/gif",
    )

with col_mp4:
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        st.warning("MP4 needs ffmpeg (`brew install ffmpeg`).")
    if st.button("Generate MP4", disabled=ffmpeg_path is None):
        mp4_progress_bar = st.progress(0.0, text="Rendering MP4 frames…")

        def _mp4_progress(current_frame, total_frames):
            frac = (current_frame + 1) / total_frames
            mp4_progress_bar.progress(
                frac, text=f"Rendering MP4 frame {current_frame + 1}/{total_frames}"
            )

        try:
            st.session_state["mp4_bytes"] = cached_render(
                render_mp4, "mp4", anim_kwargs, progress_callback=_mp4_progress
            )
            st.session_state["mp4_name"] = f"{base_name}.mp4"
            st.session_state.pop("mp4_error", None)
        except Exception as exc:
            st.session_state.pop("mp4_bytes", None)
            st.session_state["mp4_error"] = str(exc)
        finally:
            mp4_progress_bar.empty()

if err := st.session_state.get("mp4_error"):
    st.error(f"Could not render MP4: {err}")

if mp4_bytes := st.session_state.get("mp4_bytes"):
    st.video(mp4_bytes)
    st.download_button(
        "Download MP4",
        data=mp4_bytes,
        file_name=st.session_state.get("mp4_name", "transit.mp4"),
        mime="video/mp4",
    )