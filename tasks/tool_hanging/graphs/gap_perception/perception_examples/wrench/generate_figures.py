"""Regenerate explanatory figures from the saved GaP trace artifacts."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).parent


def image(name):
    return np.asarray(Image.open(ROOT / name).convert("RGB"))


def mask(name):
    return np.asarray(Image.open(ROOT / name).convert("L")) > 0


def overlay(rgb, binary, color):
    result = rgb.astype(float).copy()
    result[binary] = 0.45 * result[binary] + 0.55 * np.asarray(color)
    return result.astype(np.uint8)


def save_target_pipeline():
    rgb = image("01_overhead_rgb.png")
    wrench = mask("02_sam3_wrench_mask.png")
    ring = mask("03_sam3_ring_mask.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    items = [
        (rgb, "1. Overhead RGB-D camera\n(RGB shown)"),
        (overlay(rgb, wrench, [30, 220, 70]), "2. SAM3: adjustable wrench\nscore 0.844"),
        (overlay(rgb, ring, [255, 60, 40]), "3. SAM3: circular ring end\nscore 0.766"),
    ]
    for ax, (data, title) in zip(axes, items):
        ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(ROOT / "figure_1_wrench_and_ring_pipeline.png", dpi=180)
    plt.close(fig)


def save_hook_candidates():
    rgb = image("04_fixture_rgb.png")
    detections = json.loads((ROOT / "05_grounding_dino_hook_detections.json").read_text())["detections"]
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    ax.imshow(rgb)
    for i, detection in enumerate(detections):
        b = detection["box"]
        selected = i == 4
        color = "lime" if selected else "red"
        ax.add_patch(Rectangle((b["x1"], b["y1"]), b["x2"] - b["x1"], b["y2"] - b["y1"], fill=False, lw=3 if selected else 1.5, ec=color))
        ax.text(max(0, b["x1"]), max(10, b["y1"] - 4), f"#{i} {detection['score']:.3f}" + (" SELECTED" if selected else ""), color=color, fontsize=9, bbox={"facecolor": "black", "alpha": .65, "pad": 2})
    ax.set_title("Grounding DINO hook proposals\nred = rejected by shape/depth; green = thin depth-consistent hook")
    ax.axis("off")
    fig.savefig(ROOT / "figure_2_dino_hook_candidate_filtering.png", dpi=180)
    plt.close(fig)


def save_depth_geometry():
    ring = np.load(ROOT / "07_ring_points_world.npz")["positions"]
    roi = np.load(ROOT / "08_hook_roi_points_world.npz")["positions"]
    board_x = np.percentile(roi[:, 0], 80)
    hook = roi[roi[:, 0] < board_x - .003]
    cutoff = np.percentile(hook[:, 0], 5)
    tip_points = hook[hook[:, 0] <= cutoff]
    tip = np.median(tip_points, axis=0)
    ring_center = np.median(ring, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].scatter(ring[:, 0], ring[:, 1], s=10, alpha=.65, label="ring mask depth points")
    axes[0].scatter(ring_center[0], ring_center[1], s=140, marker="x", linewidths=3, c="red", label="coordinate-wise median")
    axes[0].set_title(f"Ring 3D center from RGB-D ({len(ring)} points)")
    axes[0].set_xlabel("world X (m)")
    axes[0].set_ylabel("world Y (m)")
    axes[0].axis("equal")
    axes[0].legend()

    axes[1].scatter(roi[:, 0], roi[:, 2], s=9, alpha=.18, c="gray", label="all DINO-box depth points")
    axes[1].scatter(hook[:, 0], hook[:, 2], s=12, alpha=.65, c="dodgerblue", label=">3 mm in front of board")
    axes[1].scatter(tip_points[:, 0], tip_points[:, 2], s=20, c="orange", label="foremost 5%")
    axes[1].scatter(tip[0], tip[2], s=150, marker="x", linewidths=3, c="red", label="median hook tip")
    axes[1].axvline(board_x, ls="--", c="black", label="80th-percentile board plane")
    axes[1].set_title(f"Hook tip from calibrated depth ({len(roi)} ROI points)")
    axes[1].set_xlabel("world X (m): toward robot is decreasing X")
    axes[1].set_ylabel("world Z (m)")
    axes[1].legend(fontsize=8)
    fig.savefig(ROOT / "figure_3_rgbd_ring_and_hook_geometry.png", dpi=180)
    plt.close(fig)


def save_feature_in_hand():
    rgb = image("09_wrist_rgb_after_grasp.png")
    ring_mask = mask("10_wrist_ring_mask.png")
    result = json.loads((ROOT / "11_ring_in_hand_result.json").read_text())["ring_in_hand"]["position"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].imshow(overlay(rgb, ring_mask, [255, 60, 40]))
    axes[0].set_title("Wrist camera after grasp\nSAM3 ring mask, score 0.680")
    axes[0].axis("off")

    axes[1].axis("off")
    axes[1].text(.5, .90, "Feature-in-hand transform", ha="center", fontsize=18, weight="bold")
    axes[1].text(.5, .72, r"$T_{hand,ring}=T_{world,hand}^{-1}T_{world,ring}$", ha="center", fontsize=19)
    axes[1].annotate("world-frame ring center\nfrom wrist mask + depth", xy=(.47, .56), xytext=(.08, .56), arrowprops={"arrowstyle": "->", "lw": 2}, ha="center")
    axes[1].annotate("current hand pose\nfrom robot.get_ee_pose", xy=(.53, .56), xytext=(.90, .56), arrowprops={"arrowstyle": "->", "lw": 2}, ha="center")
    axes[1].text(.5, .38, "Saved ring position in hand coordinates:", ha="center", fontsize=12)
    axes[1].text(.5, .27, f"x={result['x']*1000:.1f} mm,  y={result['y']*1000:.1f} mm,  z={result['z']*1000:.1f} mm", ha="center", fontsize=14, family="monospace", bbox={"facecolor": "#eaf4ff", "edgecolor": "#2878b5", "pad": 8})
    axes[1].text(.5, .08, "After the hand moves:  T_world,ring = T_world,hand · T_hand,ring", ha="center", fontsize=12)
    fig.savefig(ROOT / "figure_4_feature_in_hand_mapping.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    save_target_pipeline()
    save_hook_candidates()
    save_depth_geometry()
    save_feature_in_hand()
