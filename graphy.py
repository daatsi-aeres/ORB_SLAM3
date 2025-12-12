import matplotlib
matplotlib.use("Agg")  # headless + faster

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from datetime import datetime


# -------------------------
# Run Tag and Output Folder
# -------------------------
run_tag = "Dec11_Run1"

output_dir = f"runs/{run_tag}"
os.makedirs(output_dir, exist_ok=True)

def savefig(name):
    plt.savefig(f"{output_dir}/{name}_{run_tag}.png", dpi=300)
    plt.close()
    gc.collect()  # free memory ASAP

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
# Optimized CSV Loader
# -------------------------
def load_metrics(metric):
    base = pl.read_csv(f"baseline/{metric}.csv", try_parse_dates=False, low_memory=True)
    dip  = pl.read_csv(f"dip/{metric}.csv",      try_parse_dates=False, low_memory=True)
    return base.to_pandas(), dip.to_pandas()


# -------------------------
# 1. Map Point Culling (LINE GRAPH VERSION)
# -------------------------
df_b_raw, df_d_raw = load_metrics("map_culling_dip")   # <-- do NOT include _baseline or _dip here

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
# 3. Tracking State Timeline (Stacked)
# -------------------------
df_b, df_d = load_metrics("tracking_state_dip")

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
df_b, df_d = load_metrics("keypoints_dip")

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
df_b, df_d = load_metrics("reprojection_error_dip")

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
# df_b, df_d = load_metrics("rpe_dip")

# plt.figure(figsize=(10,5))
# plt.plot(df_b["frame_id"], df_b["rpe"], label="Baseline RPE")
# plt.plot(df_d["frame_id"], df_d["rpe"], label="DIP RPE")
# plt.title("Relative Pose Error (RPE)")
# plt.xlabel("Frame ID")
# plt.ylabel("Relative Error (meters)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# savefig("plot_rpe")


# -------------------------
# Keyframe Creation Frequency
# -------------------------
df_b, df_d = load_metrics("keyframe_frequency_dip")

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