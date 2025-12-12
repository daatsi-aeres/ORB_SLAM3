import matplotlib
matplotlib.use("Agg")  # headless + faster

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from datetime import datetime


def process_culling(raw_df):
    # Pivot CREATED/CULLED rows into columns
    pivot = raw_df.pivot_table(
        index="kf_id",
        columns="created",
        values="culled",
        aggfunc="first"
    ).reset_index()

    # Rename
    pivot = pivot.rename(columns={
        "CULLED": "culled",
        "CREATED": "created"
    }).fillna(0)

    # Compute totals
    pivot["total"] = pivot["culled"] + pivot["created"]

    # Avoid division by zero
    pivot["cull_ratio"] = pivot["culled"] / pivot["total"].replace(0, float("nan"))

    # Mean observations per keyframe
    pivot["mean_obs"] = pivot["total"] / 2

    return pivot


# -------------------------
# Auto-generated Run Tag
# -------------------------
# Format example: 2025-02-14_12-18PM
now = datetime.now()
run_tag = now.strftime("%Y-%m-%d_%I-%M%p")   # auto-format date + time

output_dir = f"runs/{run_tag}"
os.makedirs(output_dir, exist_ok=True)

print(f"Run tag generated: {run_tag}")


output_dir = f"runs/{run_tag}"
os.makedirs(output_dir, exist_ok=True)

def savefig(name):
    plt.savefig(f"{output_dir}/{name}_{run_tag}.png", dpi=300)
    plt.close()
    gc.collect()  # free memory ASAP


# -------------------------
# Optimized CSV Loader
# -------------------------
def load_metrics(metric):
    base = pl.read_csv(f"baseline/{metric}.csv", try_parse_dates=False, low_memory=True)
    dip  = pl.read_csv(f"dip/{metric}.csv",      try_parse_dates=False, low_memory=True)
    return base.to_pandas(), dip.to_pandas()


# -------------------------
# 1. Map Point Culling (LINE GRAPH VERSION)
# -------------------------
df_b_raw, df_d_raw = load_metrics("map_culling")   # <-- do NOT include _baseline or _dip here

# Process raw CREATED/CULLED rows
df_b = process_culling(df_b_raw)
df_d = process_culling(df_d_raw)

plt.figure(figsize=(12,5))

# Line plot for culling ratio
plt.plot(df_b["kf_id"], df_b["cull_ratio"], label="Baseline Cull Ratio", linewidth=2)
plt.plot(df_d["kf_id"], df_d["cull_ratio"], label="DIP Cull Ratio", linewidth=2)

plt.title("Map Point Cull Ratio per Keyframe (Line Graph)")
plt.xlabel("Keyframe ID")
plt.ylabel("Cull Ratio")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
savefig("plot_map_culling_line")
plt.close()

df_b.to_csv(f"{output_dir}/map_culling_processed_baseline_{run_tag}.csv", index=False)
df_d.to_csv(f"{output_dir}/map_culling_processed_dip_{run_tag}.csv", index=False)

# -------------------------
# Map Point Created vs Culled (Baseline)
# -------------------------
fig = plt.figure(figsize=(12,6))   # <-- use fig, not just plt.figure

plt.plot(
    df_b["kf_id"],
    df_b["created"],
    label="Created Map Points",
    linewidth=2,
    color="tab:green"
)

plt.plot(
    df_b["kf_id"],
    df_b["culled"],
    label="Culled Map Points",
    linewidth=2,
    color="tab:red"
)

plt.title("Baseline: Map Point Creation vs Culling per Keyframe")
plt.xlabel("Keyframe ID")
plt.ylabel("Number of Map Points")
plt.grid(True)
plt.legend()

# ---- Caption BELOW the plot ----
fig.text(
    0.5, -0.12,
    "Interpretation: Peaks in culled map points indicate visually degraded frames "
    "(e.g., motion blur, poor lighting, or dynamic objects), where newly created "
    "features fail to remain stable. Sustained map point creation with low culling "
    "suggests robust feature detection and stable tracking.",
    ha="center",
    fontsize=10,
    wrap=True
)

plt.tight_layout()
savefig("plot_map_points_created_vs_culled_baseline")


# -------------------------
# Map Point Created vs Culled (DIP)
# -------------------------
fig = plt.figure(figsize=(12,6))   # <-- again, use fig

plt.plot(
    df_d["kf_id"],
    df_d["created"],
    label="Created Map Points",
    linewidth=2,
    color="tab:green"
)

plt.plot(
    df_d["kf_id"],
    df_d["culled"],
    label="Culled Map Points",
    linewidth=2,
    color="tab:red"
)

plt.title("DIP: Map Point Creation vs Culling per Keyframe")
plt.xlabel("Keyframe ID")
plt.ylabel("Number of Map Points")
plt.grid(True)
plt.legend()

# ---- Caption BELOW the plot ----
fig.text(
    0.5, -0.12,
    "Interpretation: Relative reductions in culling peaks compared to the baseline "
    "indicate improved image quality from the DIP pipeline, resulting in more "
    "persistent map points and increased robustness under challenging visual conditions.",
    ha="center",
    fontsize=10,
    wrap=True
)

plt.tight_layout()
savefig("plot_map_points_created_vs_culled_dip")

# -------------------------
# 3. Tracking State Timeline (Stacked)
# -------------------------
df_b, df_d = load_metrics("tracking_state")

fig, axes = plt.subplots(2, 1, figsize=(14,6), sharex=True)

def add_stats_box(ax, df, title_color="black"):
    # Count occurrences of each tracking state
    ok_count = (df["state"] == 2).sum()
    recent_lost_count = (df["state"] == 3).sum()
    lost_count = (df["state"] == 4).sum()
    
    text = (
        f"OK: {ok_count}\n"
        f"Recently Lost: {recent_lost_count}\n"
        f"LOST: {lost_count}"
    )

    # Add the annotation in bottom-right corner
    ax.text(
        0.98, 0.02, text,
        transform=ax.transAxes,
        fontsize=10,
        va='bottom',
        ha='right',
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc="white",
            ec=title_color,
            alpha=0.8
        )
    )

# --- Baseline ---
axes[0].scatter(df_b["frame_id"], df_b["state"], c="blue", s=6)
axes[0].set_title("Tracking State – Baseline")
axes[0].set_yticks([-1,0,1,2,3,4,5])
axes[0].set_yticklabels(["SYS_NOT_RDY","NO_IMG_YET","NOT_INIT","OK", "RECENTLY_LOST", "LOST","OK_KLT"])
axes[0].grid(True)
add_stats_box(axes[0], df_b, title_color="blue")

# --- DIP ---
axes[1].scatter(df_d["frame_id"], df_d["state"], c="orange", s=6)
axes[1].set_title("Tracking State – DIP")
axes[1].set_yticks([-1,0,1,2,3,4,5])
axes[1].set_yticklabels(["SYS_NOT_RDY","NO_IMG_YET","NOT_INIT","OK", "RECENTLY_LOST", "LOST","OK_KLT"])
axes[1].set_xlabel("Frame ID")
axes[1].grid(True)
add_stats_box(axes[1], df_d, title_color="orange")

plt.tight_layout()
savefig("plot_tracking_state_stacked")


# -------------------------
# 4. Keypoints (Stacked Baseline + DIP)
# -------------------------
df_b, df_d = load_metrics("keypoints")

fig, axes = plt.subplots(2, 1, figsize=(12,7), sharex=True)

axes[0].plot(df_b["frame_id"], df_b["keypoints"], label="Keypoints", alpha=0.7)
axes[0].plot(df_b["frame_id"], df_b["tracked"], label="Tracked Map Points", alpha=0.7)
axes[0].set_title("Baseline: Keypoints & Tracked Map Points")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df_d["frame_id"], df_d["keypoints"], label="Keypoints", alpha=0.7)
axes[1].plot(df_d["frame_id"], df_d["tracked"], label="Tracked Map Points", alpha=0.7)
axes[1].set_title("DIP: Keypoints & Tracked Map Points")
axes[1].set_ylabel("Count")
axes[1].set_xlabel("Frame ID")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
savefig("plot_keypoints_stacked")


# -------------------------
# Reprojection Error
# -------------------------
df_b, df_d = load_metrics("reprojection_error")

plt.figure(figsize=(10,5))
plt.plot(df_b["frame_id"], df_b["reproj_mean"], label="Baseline Mean")
plt.plot(df_d["frame_id"], df_d["reproj_mean"], label="DIP Mean")
plt.plot(df_b["frame_id"], df_b["reproj_std"], '--', label="Baseline Std")
plt.plot(df_d["frame_id"], df_d["reproj_std"], '--', label="DIP Std")
plt.title("Reprojection Error (Mean & Std)")
plt.xlabel("Frame ID")
plt.ylabel("Error (pixels)")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig("plot_reprojection_error")

# -------------------------
# Relative Pose Error (RPE)
# -------------------------
#df_b, df_d = load_metrics("rpe")

#plt.figure(figsize=(10,5))
#plt.plot(df_b["frame_id"], df_b["rpe"], label="Baseline RPE")
#lt.plot(df_d["frame_id"], df_d["rpe"], label="DIP RPE")
#plt.title("Relative Pose Error (RPE)")
#plt.xlabel("Frame ID")
#plt.ylabel("Relative Error (meters)")
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#savefig("plot_rpe")

# -------------------------
# Keyframe Creation Frequency
# -------------------------
df_b, df_d = load_metrics("keyframe_frequency")

def compute_kf_intervals(df):
    df = df.sort_values("kf_id")
    df["frame_delta"] = df["frame_id"].diff().fillna(0)
    if "timestamp" in df.columns:
        df["time_delta"] = df["timestamp"].diff().fillna(0)
    else:
        df["time_delta"] = 0
    return df

df_b = compute_kf_intervals(df_b)
df_d = compute_kf_intervals(df_d)

plt.figure(figsize=(10,5))
plt.plot(df_b["kf_id"], df_b["frame_delta"], label="Baseline", marker="o")
plt.plot(df_d["kf_id"], df_d["frame_delta"], label="DIP", marker="o")

plt.title("Keyframe Creation Frequency (Frames Between Keyframes)")
plt.xlabel("Keyframe ID")
plt.ylabel("Frames Between Keyframes")
plt.grid(True)
plt.legend()
plt.tight_layout()
savefig("plot_keyframe_frequency")


# -------------------------
# Camera Pose Jump Magnitude
# -------------------------

def load_trajectory_txt(path):
    """
    Load ORB-SLAM3 trajectory in TUM format:
    timestamp tx ty tz qx qy qz qw
    """
    cols = ["timestamp","tx","ty","tz","qx","qy","qz","qw"]
    try:
        df = pl.read_csv(path, separator=" ", has_header=False, new_columns=cols)
    except:
        # fallback if PL can't auto-detect space separation
        df = pl.read_csv(path, has_header=False, new_columns=cols)
    return df.to_pandas()


def compute_pose_jumps(df):
    """
    Computes translation jump per frame:
    || position[i] - position[i-1] ||
    """
    pos = df[["tx","ty","tz"]].values
    # difference between consecutive positions
    diffs = np.diff(pos, axis=0)
    # Euclidean norm of each diff
    jumps = np.linalg.norm(diffs, axis=1)
    return jumps


# Load baseline and DIP trajectory text files
cam_base = load_trajectory_txt("baseline/CameraTrajectory.txt")
cam_dip  = load_trajectory_txt("dip/CameraTrajectory.txt")

# Compute pose jumps
jumps_base = compute_pose_jumps(cam_base)
jumps_dip  = compute_pose_jumps(cam_dip)

# --- Plot ---
plt.figure(figsize=(12,5))
plt.plot(jumps_base, label="Baseline Pose Jump Magnitude", linewidth=1.5)
plt.plot(jumps_dip,  label="DIP Pose Jump Magnitude", linewidth=1.5)

plt.title("Camera Pose Jump Magnitude Per Frame")
plt.xlabel("Frame Index")
plt.ylabel("Jump Magnitude (meters)")
plt.grid(True)
plt.legend()
plt.tight_layout()
savefig("plot_camera_pose_jump_magnitude")
plt.close()