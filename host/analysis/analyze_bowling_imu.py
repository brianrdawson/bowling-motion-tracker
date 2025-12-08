import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

# ---- Tunable constants ----
GYRO_THRESH = 5.0        # dps threshold for "motion happening"
MIN_THROW_SAMPLES = 3    # minimum samples above threshold to count as a throw
PLOT_DPI = 200

# Approx distance from elbow/shoulder pivot to wrist IMU (meters)
ARM_LENGTH_M = 0.7       # TODO: adjust if needed


def load_and_clean(csv_path: str) -> pd.DataFrame:
    print(f"Loading: {csv_path}")

    df = pd.read_csv(
        csv_path,
        comment="#",
        names=[
            "t_ms",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "accel_mag", "gyro_mag",
        ]
    )

    df = df.dropna(how="all")

    # Convert numeric columns
    cols_to_convert = ["t_ms", "ax", "ay", "az", "gx", "gy", "gz", "accel_mag", "gyro_mag"]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=cols_to_convert)

    # Time in seconds
    df["t"] = df["t_ms"] / 1000.0

    # Compute swing angle from accel + gravity reference
    df = compute_swing_angle(df)

    return df


def compute_swing_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate a 'swing angle' (deg) from the accelerometer direction relative
    to a reference gravity vector (taken from the lowest-motion samples).
    """
    if df.empty:
        df["swing_angle_deg"] = np.nan
        return df

    # Use the N samples with smallest gyro magnitude as "quiet" = gravity reference
    n_ref = min(50, len(df))
    quiet = df.nsmallest(n_ref, "gyro_mag")
    ref_vec = quiet[["ax", "ay", "az"]].mean().to_numpy()

    ref_norm = np.linalg.norm(ref_vec)
    if ref_norm == 0:
        df["swing_angle_deg"] = np.nan
        return df

    ref_unit = ref_vec / ref_norm

    accel = df[["ax", "ay", "az"]].to_numpy()
    accel_norm = np.linalg.norm(accel, axis=1, keepdims=True)
    accel_norm[accel_norm == 0] = 1.0  # avoid div by zero
    accel_unit = accel / accel_norm

    # cos(theta) between each accel vector and the reference
    cos_theta = np.clip(accel_unit @ ref_unit, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)

    df["swing_angle_deg"] = theta_deg
    return df


def detect_throws(df: pd.DataFrame, gyro_thresh: float, min_samples: int):
    """
    Return a list of (throw_id, df_segment) for each detected throw.
    Detection: gyro_mag > threshold for consecutive samples.
    """
    active = df["gyro_mag"] > gyro_thresh

    throw_ids = []
    current_id = 0
    in_throw = False

    for is_active in active:
        if is_active and not in_throw:
            in_throw = True
            current_id += 1
        elif not is_active and in_throw:
            in_throw = False
        throw_ids.append(current_id if in_throw else 0)

    df["throw_id"] = throw_ids

    throws = []
    for tid in sorted(set(df["throw_id"])):
        if tid == 0:
            continue
        seg = df[df["throw_id"] == tid]
        if len(seg) >= min_samples:
            throws.append((tid, seg.copy()))

    return throws


def estimate_wrist_rotation(seg: pd.DataFrame, release_time: float, window_s: float = 0.05):
    """
    Estimate wrist rotation angle (deg) around release.
    We integrate the gyro axis that has the highest variance near release.
    """
    win = seg[(seg["t"] >= release_time - window_s) & (seg["t"] <= release_time + window_s)]
    if len(win) < 2:
        return 0.0, None  # not enough data

    axis_candidates = ["gx", "gy", "gz"]
    axis = max(axis_candidates, key=lambda c: win[c].var())

    dt = win["t"].diff().fillna(0.0)
    rotation_deg = (win[axis] * dt).sum()          # signed rotation (unused for now)
    rotation_deg_abs = (win[axis].abs() * dt).sum()

    return rotation_deg_abs, axis


def swing_extrema(seg: pd.DataFrame):
    """
    Find backswing top, lowest point, and forward-swing top within a throw segment
    using swing_angle_deg.

    Returns dict with times and angles, or None if segment too short.
    """
    if "swing_angle_deg" not in seg or len(seg) < 3:
        return None

    # Max angle before everything else -> backswing top
    idx_back = seg["swing_angle_deg"].idxmax()

    # Lowest angle after backswing -> bottom of swing
    seg_after_back = seg.loc[idx_back:]
    idx_low = seg_after_back["swing_angle_deg"].idxmin()

    # Max angle after low -> follow-through top
    seg_after_low = seg_after_back.loc[idx_low:]
    idx_forward = seg_after_low["swing_angle_deg"].idxmax()

    return {
        "back_top_t":      seg.loc[idx_back, "t"],
        "back_top_angle":  seg.loc[idx_back, "swing_angle_deg"],
        "low_t":           seg.loc[idx_low, "t"],
        "low_angle":       seg.loc[idx_low, "swing_angle_deg"],
        "forward_top_t":   seg.loc[idx_forward, "t"],
        "forward_top_angle": seg.loc[idx_forward, "swing_angle_deg"],
    }

def animate_throw_arc(seg: pd.DataFrame,
                      arm_length_m: float,
                      base_path: Path,
                      throw_id: int,
                      save_gif: bool = True,
                      save_mp4: bool = False):
    """
    Create an animated 2D arm arc for a single throw.
    Saves GIF and/or MP4 showing the arm moving along the arc.
    """

    if "swing_angle_deg" not in seg or seg["swing_angle_deg"].isna().all():
        return

    # Use same geometry as static arc (with flipped X)
    theta = np.radians(seg["swing_angle_deg"].to_numpy())
    theta0 = theta.min()
    theta_rel = theta - theta0

    x = -arm_length_m * np.sin(theta_rel)
    y = -arm_length_m * np.cos(theta_rel)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Forward / Back (m)")
    ax.set_ylabel("Up / Down (m)")
    ax.set_title(f"Throw {throw_id} – Arm Arc Animation")
    ax.grid(True, alpha=0.3)

    # Pre-set limits so the view doesn't jump
    margin = arm_length_m * 0.1
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)

    # Static path in light gray
    ax.plot(x, y, linewidth=1.0, alpha=0.3)

    # Moving marker (the “wrist”)
    point, = ax.plot([], [], marker="o", markersize=8)

    # Optional trace of path so far
    trace_line, = ax.plot([], [], linewidth=2)

    def init():
        point.set_data([], [])
        trace_line.set_data([], [])
        return point, trace_line

    def update(frame):
        # frame is an index into x/y
        point.set_data([x[frame]], [y[frame]])
        trace_line.set_data(x[:frame+1], y[:frame+1])
        return point, trace_line

    frames = len(x)
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=30,    # ms between frames (~33 fps)
        blit=True
    )

    base = base_path.with_suffix(f".throw{throw_id}_arc")

    if save_gif:
        gif_path = base.with_suffix(".gif")
        anim.save(gif_path, writer=animation.PillowWriter(fps=30))
        print(f"Saved arc animation (GIF) for throw {throw_id} to: {gif_path}")

    if save_mp4:
        mp4_path = base.with_suffix(".mp4")
        anim.save(mp4_path, writer="ffmpeg", fps=30)
        print(f"Saved arc animation (MP4) for throw {throw_id} to: {mp4_path}")

    plt.close(fig)

def plot_throw_arc(seg: pd.DataFrame, arm_length_m: float, base_path: Path, throw_id: int):
    """
    Plot a 2D arm arc with backswing (orange) and forward swing (blue) color-coded.
    Uses swing_angle_deg and extrema indices.
    """

    # Must have swing angle
    if "swing_angle_deg" not in seg or seg["swing_angle_deg"].isna().all():
        return

    # Get swing extrema (backswing top, low point, forward swing top)
    extrema = swing_extrema(seg)
    if extrema is None:
        return

    # Convert extrema times -> indices for segmentation
    idx_back = seg.index.get_loc(seg["t"].sub(extrema["back_top_t"]).abs().idxmin())
    idx_low = seg.index.get_loc(seg["t"].sub(extrema["low_t"]).abs().idxmin())
    idx_forward = seg.index.get_loc(seg["t"].sub(extrema["forward_top_t"]).abs().idxmin())

    theta = np.radians(seg["swing_angle_deg"].to_numpy())
    theta0 = theta.min()
    theta_rel = theta - theta0

    # Flip x so backswing appears on LEFT
    x = -arm_length_m * np.sin(theta_rel)
    y = -arm_length_m * np.cos(theta_rel)

    # Segment indices
    xs_back = x[:idx_back+1]
    ys_back = y[:idx_back+1]

    xs_down = x[idx_back:idx_low+1]
    ys_down = y[idx_back:idx_low+1]

    xs_forward = x[idx_low:idx_forward+1]
    ys_forward = y[idx_low:idx_forward+1]

    fig, ax = plt.subplots(figsize=(5, 4))

    # Draw arcs
    ax.plot(xs_back, ys_back, color="orange", linewidth=2.5, label="Backswing")
    ax.plot(xs_down, ys_down, color="gray", alpha=0.5, linewidth=2)
    ax.plot(xs_forward, ys_forward, color="blue", linewidth=2.5, label="Forward Swing")

    # Mark the three extrema points
    ax.scatter([xs_back[-1]], [ys_back[-1]], color="orange", s=60, marker="^")     # top of backswing
    ax.scatter([xs_down[-1]], [ys_down[-1]], color="red", s=60, marker="v")        # bottom/low point
    ax.scatter([xs_forward[-1]], [ys_forward[-1]], color="blue", s=60, marker="^") # top of follow-through

    ax.set_aspect("equal", "box")
    ax.set_xlabel("Forward / Back (m)")
    ax.set_ylabel("Up / Down (m)")
    ax.set_title(f"Throw {throw_id} – Arm Arc (Color Coded)")

    ax.grid(True, alpha=0.3)
    ax.legend()

    arc_path = base_path.with_suffix(f".throw{throw_id}_arc.png")
    fig.tight_layout()
    fig.savefig(arc_path, dpi=200)
    plt.close(fig)

    print(f"Saved color-coded arc plot for throw {throw_id} to: {arc_path}")

def animate_throw_arc_basic(
    seg: pd.DataFrame,
    arm_length_m: float,
    base_path: Path,
    throw_id: int,
    seconds_per_point: float = 0.5,
    save_gif: bool = True,
    save_mp4: bool = False,
):
    """
    Simple 2D animation of a single throw:
      - Arc in 2D
      - Point moves slowly along arc
      - Backswing frames = green
      - Forward swing frames = blue
      - Release is clearly marked in red
    """

    # Need swing angle and extrema
    if "swing_angle_deg" not in seg or seg["swing_angle_deg"].isna().all():
        return

    extrema = swing_extrema(seg)
    if extrema is None:
        return

    # Convert extrema times to indices (for phase split)
    idx_back = seg.index.get_loc(seg["t"].sub(extrema["back_top_t"]).abs().idxmin())
    idx_low = seg.index.get_loc(seg["t"].sub(extrema["low_t"]).abs().idxmin())
    idx_forward = seg.index.get_loc(seg["t"].sub(extrema["forward_top_t"]).abs().idxmin())

    # Release = max gyro_mag in this throw
    release_idx_global = seg["gyro_mag"].idxmax()
    release_frame = seg.index.get_loc(release_idx_global)

    theta = np.radians(seg["swing_angle_deg"].to_numpy())
    theta0 = theta.min()
    theta_rel = theta - theta0

    # Flip X so backswing is on the LEFT, forward swing on the RIGHT
    x = -arm_length_m * np.sin(theta_rel)
    y = -arm_length_m * np.cos(theta_rel)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Forward / Back (m)")
    ax.set_ylabel("Up / Down (m)")
    ax.set_title(f"Throw {throw_id} – Basic 2D Swing Animation")
    ax.grid(True, alpha=0.3)

    margin = arm_length_m * 0.1
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)

    # Draw the full arc lightly in gray
    ax.plot(x, y, color="lightgray", linewidth=1.0)

    # Static marker at release location (red X)
    release_x = x[release_frame]
    release_y = y[release_frame]
    release_mark = ax.scatter(
        [release_x],
        [release_y],
        color="red",
        marker="x",
        s=70,
        alpha=0.8,
        label="Release"
    )

    # Moving point + trace
    point, = ax.plot([], [], marker="o", markersize=8)
    trace, = ax.plot([], [], linewidth=2)

    frames = len(x)
    fps = 1.0 / seconds_per_point  # points per second

    def init():
        point.set_data([], [])
        trace.set_data([], [])
        return point, trace, release_mark

    def update(frame_idx):
        # Base color by phase
        if frame_idx <= idx_low:
            base_color = "green"   # backswing/down to low
        else:
            base_color = "blue"    # forward swing / follow-through

        color = base_color
        size = 8

        # Highlight release frame
        if frame_idx == release_frame:
            color = "red"
            size = 11

        point.set_data([x[frame_idx]], [y[frame_idx]])
        point.set_markerfacecolor(color)
        point.set_markeredgecolor(color)
        point.set_markersize(size)

        trace.set_data(x[:frame_idx + 1], y[:frame_idx + 1])
        trace.set_color(base_color)

        return point, trace, release_mark

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=seconds_per_point * 1000,  # ms per frame
        blit=True,
        repeat=False,
    )

    stem = base_path.stem  # "mock_imu_session"
    parent = base_path.parent

    if save_gif:
        gif_path = parent / f"{stem}.throw{throw_id}_arc_basic.gif"
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
        print(f"Saved basic 2D swing GIF for throw {throw_id} to: {gif_path}")

    if save_mp4:
        mp4_path = parent / f"{stem}.throw{throw_id}_arc_basic.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=fps)
        print(f"Saved basic 2D swing MP4 for throw {throw_id} to: {mp4_path}")


    plt.close(fig)

def analyze(csv_path: str,
            gyro_thresh: float = GYRO_THRESH,
            min_samples: int = MIN_THROW_SAMPLES,
            arm_length_m: float = ARM_LENGTH_M):
    df = load_and_clean(csv_path)

    # Detect throws
    throws = detect_throws(df, gyro_thresh, min_samples)

    if not throws:
        print("No throws detected. Try lowering GYRO_THRESH.")
        return

    print(f"Detected {len(throws)} throw(s).\n")

    summary_rows = []
    release_times = []
    release_values = []

    for tid, seg in throws:
        # Release = max gyro_mag in this throw
        rel_idx = seg["gyro_mag"].idxmax()
        rel_row = df.loc[rel_idx]

        t_start = seg["t"].iloc[0]
        t_end = seg["t"].iloc[-1]
        duration = t_end - t_start
        release_time = rel_row["t"]
        release_gyro_dps = rel_row["gyro_mag"]

        # Arm / ball speed estimate
        omega_rad_s = release_gyro_dps * math.pi / 180.0
        wrist_speed_mps = omega_rad_s * arm_length_m
        wrist_speed_mph = wrist_speed_mps * 2.23694

        # Peak accel for reference
        peak_accel = seg["accel_mag"].max()

        # Wrist rotation near release
        wrist_rot_deg, rot_axis = estimate_wrist_rotation(seg, release_time, window_s=0.05)

        # Swing angle extrema (backswing / bottom / forward top)
        extrema = swing_extrema(seg)
        if extrema is None:
            extrema = {
                "back_top_t": None,
                "back_top_angle": None,
                "low_t": None,
                "low_angle": None,
                "forward_top_t": None,
                "forward_top_angle": None,
            }

        summary_rows.append({
            "throw_id": tid,
            "t_start_s": t_start,
            "t_end_s": t_end,
            "duration_s": duration,
            "release_time_s": release_time,
            "release_gyro_dps": release_gyro_dps,
            "wrist_speed_mph": wrist_speed_mph,
            "peak_accel_g": peak_accel,
            "wrist_rot_deg_±0.05s": wrist_rot_deg,
            "wrist_rot_axis": rot_axis,
            "back_top_t": extrema["back_top_t"],
            "back_top_angle": extrema["back_top_angle"],
            "low_t": extrema["low_t"],
            "low_angle": extrema["low_angle"],
            "forward_top_t": extrema["forward_top_t"],
            "forward_top_angle": extrema["forward_top_angle"],
        })

        release_times.append(release_time)
        release_values.append(release_gyro_dps)

    summary_df = pd.DataFrame(summary_rows)
    print("Per-throw summary (approximate):")
    print(summary_df.to_string(index=False))
    print(f"\nAssumed arm length for speed calc: {arm_length_m:.2f} m\n")

    # ---- Plot and save PNG ----
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot accel + gyro
    ax.plot(df["t"], df["accel_mag"], label="Accel Magnitude (g)")
    ax.plot(df["t"], df["gyro_mag"], label="Gyro Magnitude (dps)")

    # Mark release points
    ax.scatter(release_times, release_values, s=80, label="Release (max gyro)", zorder=5)

    # Per-throw arc plots (backswing → low → forward swing)
    base_path = Path(csv_path)
 
    # Shade throw regions
    for tid, seg in throws:
        ax.axvspan(seg["t"].iloc[0], seg["t"].iloc[-1], alpha=0.08)
        plot_throw_arc(seg, arm_length_m, base_path, tid)
        animate_throw_arc_basic(
            seg,
            arm_length_m,
            base_path,
            tid,
            seconds_per_point=1.0,   # or 1.0 for very slow
            save_gif=True,
            save_mp4=False,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accel / Gyro Magnitude")
    ax.grid(True)

    # Second y-axis for swing angle
    ax2 = ax.twinx()
    ax2.plot(df["t"], df["swing_angle_deg"], alpha=0.4, label="Swing Angle (deg)", color="tab:green")
    ax2.set_ylabel("Swing Angle (deg)")

    # Mark backswing / low / forward-top points
    for row in summary_rows:
        if row["back_top_t"] is not None:
            ax2.scatter(row["back_top_t"], row["back_top_angle"], marker="^", s=60, color="blue")
        if row["low_t"] is not None:
            ax2.scatter(row["low_t"], row["low_angle"], marker="v", s=60, color="red")
        if row["forward_top_t"] is not None:
            ax2.scatter(row["forward_top_t"], row["forward_top_angle"], marker="^", s=60, color="purple")

    fig.suptitle("Bowling Wrist IMU – Multi-Throw Trace + Swing Angle")
    fig.tight_layout()

    csv_path = Path(csv_path)
    png_path = csv_path.with_suffix(".plot.png")
    fig.savefig(png_path, dpi=PLOT_DPI)
    print(f"Saved plot to: {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Bowling IMU CSV Data (multi-throw, speed, rotation, swing angle)"
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help="Path to the CSV file captured from the XIAO logger"
    )
    parser.add_argument(
        "--gyro_thresh",
        type=float,
        default=GYRO_THRESH,
        help=f"Gyro magnitude threshold (dps) for throw detection (default={GYRO_THRESH})"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=MIN_THROW_SAMPLES,
        help=f"Minimum samples above threshold to count as a throw (default={MIN_THROW_SAMPLES})"
    )
    parser.add_argument(
        "--arm_length_m",
        type=float,
        default=ARM_LENGTH_M,
        help=f"Approx arm length in meters for speed estimate (default={ARM_LENGTH_M})"
    )

    args = parser.parse_args()
    print("Parsed Arguments:", args)

    analyze(
        csv_path=args.file_path,
        gyro_thresh=args.gyro_thresh,
        min_samples=args.min_samples,
        arm_length_m=args.arm_length_m,
    )
