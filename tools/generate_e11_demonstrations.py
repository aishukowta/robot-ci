import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
repo_root = Path(r"C:\Users\Aishwarya Kowta\Desktop\robot-ci")
sys.path.insert(0, str(repo_root))

from robot_ci.writing.analysis import detect_page, _page_orderings, rectify, extract_ink, PAGE_PX_PER_MM
from robot_ci.writing.plan import PenPlan
from robot_ci.writing.targets import get_target
from robot_ci.writing.metrics import strokes_to_points, nearest_distances, trajectory_error, writing_accuracy
from robot_ci.writing.simulate import make_writing_simulator
from tools.e11_scale_corrected_reanalysis import fit_similarity_icp

out_dir = repo_root / "docs/figures/E11_demonstration"
out_dir.mkdir(parents=True, exist_ok=True)

PLANS = [
    {
        "id": 1,
        "name": "plan1_hello_baseline",
        "title": "Plan 1: word_hello (writing_baseline)",
        "photo": "lab_recordings/plan1_hello_baseline.jpeg",
        "hash": "4b69978ebb27afd1",
        "target": "word_hello",
        "policy": "writing_baseline",
        "defect_note": "Nominal baseline policy. Physical ink matches intended stroke structure with 60% accuracy due to single-stroke O closure gap.",
    },
    {
        "id": 2,
        "name": "plan2_hello_candidate",
        "title": "Plan 2: word_hello (writing_candidate - star defect)",
        "photo": "lab_recordings/plan2_hello_candidate.jpeg",
        "hash": "9b7a63446cfded6b",
        "target": "word_hello",
        "policy": "writing_candidate",
        "defect_note": "Injected defect: internal star flourish inside letter 'H'. Clear red deviation (>2mm error) visible in star strokes.",
    },
    {
        "id": 3,
        "name": "plan3_square_baseline",
        "title": "Plan 3: pattern_square (writing_baseline)",
        "photo": "lab_recordings/plan3_square_baseline.jpeg",
        "hash": "f9d9e3adaa619f3e",
        "target": "pattern_square",
        "policy": "writing_baseline",
        "defect_note": "Nominal square baseline policy. 0.37 mm mean trajectory error with 97.1% coverage at 1mm.",
    },
    {
        "id": 4,
        "name": "plan4_square_candidate",
        "title": "Plan 4: pattern_square (writing_candidate - corner defect)",
        "photo": "lab_recordings/plan4_square_candidate.jpeg",
        "hash": "2ab05ca5e98d11c6",
        "target": "pattern_square",
        "policy": "writing_candidate",
        "defect_note": "Injected defect: chamfered / cut corners and top-right pen lift overshoot. Clear red deviation (>2mm error) at all 4 corners.",
    },
]

metadata = {}

for p in PLANS:
    plan_path = list((repo_root / "results/writing/exports").glob(f"*{p['hash']}*"))[0] / "plan.json"
    plan = PenPlan.load(plan_path)
    target = get_target(plan.target_id, repo_root / "writing_targets")
    layout = target.layout()
    W, H = plan.page_width_mm, plan.page_height_mm
    intended_pts = strokes_to_points(layout.strokes, 0.25)
    
    # 1. Raw photo & Page Detection
    raw_img = cv2.imread(str(repo_root / p["photo"]))
    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    det = detect_page(raw_img, W, H)
    
    annotated_raw = raw_rgb.copy()
    pts = det.corners_px.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated_raw, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
    for pt in det.corners_px:
        cv2.circle(annotated_raw, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), -1)
    
    # Save standalone annotated lab photo
    fig_raw, ax_raw = plt.subplots(figsize=(8, 6), dpi=150)
    ax_raw.imshow(annotated_raw)
    ax_raw.set_title(f"Lab Recording Photo with Page Boundary: {Path(p['photo']).name}", fontsize=11, fontweight="bold")
    ax_raw.axis("off")
    plt.tight_layout()
    out_raw = out_dir / f"{p['name']}_annotated_lab_photo.png"
    fig_raw.savefig(str(out_raw), bbox_inches="tight")
    plt.close(fig_raw)
    
    # 2. Rectify & Extract Ink
    orderings = _page_orderings(det.corners_px, H >= W)
    best_page = None
    for corners, ord_name in orderings:
        warped, Hm = rectify(raw_img, corners, W, H)
        ink = extract_ink(warped, PAGE_PX_PER_MM, "dark")
        score = float(nearest_distances(ink["points_mm"], intended_pts).mean()) if len(ink["points_mm"]) else np.inf
        if best_page is None or score < best_page[0]:
            best_page = (score, corners, ord_name, warped, Hm, ink)
    _, corners, ord_name, warped, Hm, ink = best_page
    real_pts = ink["points_mm"]
    
    # 3. PyBullet Simulation Rollout
    sim = make_writing_simulator("pybullet")
    try:
        sim_exec = sim.execute(plan, plan.seed)
    finally:
        if hasattr(sim, "close"):
            sim.close()
    sim_strokes = sim_exec.ink_strokes()
    sim_pts = strokes_to_points(sim_strokes, 0.25)
    
    # 4. Corrected real points (Scale & Orientation calibrated via Similarity ICP)
    fit = fit_similarity_icp(real_pts, intended_pts)
    corrected_real = fit["corrected_points"]
    d_real_to_intended = nearest_distances(corrected_real, intended_pts)
    
    te = trajectory_error(corrected_real, layout.strokes)
    acc = writing_accuracy(corrected_real, layout)
    d_ref = nearest_distances(intended_pts, corrected_real)
    cov1 = float((d_ref <= 1.0).mean())
    
    # Bounding box calculation for zoomed-out view with healthy margins
    all_x = list(intended_pts[:, 0]) + list(sim_pts[:, 0]) + list(corrected_real[:, 0])
    all_y = list(intended_pts[:, 1]) + list(sim_pts[:, 1]) + list(corrected_real[:, 1])
    margin = 12.0  # comfortable zoom-out margin in mm
    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin
    
    # -------------------------------------------------------------
    # FIGURE 1: 4-PANEL DEMONSTRATION COMPARISON
    # 1. Lab photo (lab_recordings)
    # 2. PyBullet Simulation (Clean, no overlap)
    # 3. Real Physical Robot Drawing (Clean, no overlap, scaled & upright)
    # 4. Difference & Deviation Map (Color-coded error & diff metrics)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=150)
    
    # Panel 1: Lab Recording Photo
    axes[0].imshow(annotated_raw)
    axes[0].set_title(f"1. Lab Recording Photo\n({Path(p['photo']).name})", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    # Panel 2: Simulation (PyBullet)
    axes[1].set_facecolor("#fafafa")
    for i, s in enumerate(layout.strokes):
        axes[1].plot(s[:, 0], s[:, 1], color="#e0e0e0", lw=3.0, linestyle="--", label="Intended font" if i == 0 else None)
    for i, s in enumerate(sim_strokes):
        axes[1].plot(s[:, 0], s[:, 1], color="#1f77b4", lw=2.2, label="PyBullet Trajectory" if i == 0 else None)
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymax, ymin)
    axes[1].set_aspect("equal")
    axes[1].set_title("2. Simulation (PyBullet)\n(No overlap: clean reference)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Page X (mm)")
    axes[1].set_ylabel("Page Y (mm)")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    
    # Panel 3: Real Physical Robot Drawing
    axes[2].set_facecolor("#fafafa")
    for i, s in enumerate(layout.strokes):
        axes[2].plot(s[:, 0], s[:, 1], color="#e0e0e0", lw=3.0, linestyle="--", label="Intended font" if i == 0 else None)
    axes[2].scatter(corrected_real[:, 0], corrected_real[:, 1], color="#111111", s=3, alpha=0.85,
                    label=f"Real Ink ({fit['scale_factor']:.2f}x, {fit['rotation_deg']:.1f}°)")
    axes[2].set_xlim(xmin, xmax)
    axes[2].set_ylim(ymax, ymin)
    axes[2].set_aspect("equal")
    axes[2].set_title("3. Physical Robot Drawing\n(Upright & scaled, clean ink)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Page X (mm)")
    axes[2].set_ylabel("Page Y (mm)")
    axes[2].legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[2].grid(True, linestyle="--", alpha=0.3)
    
    # Panel 4: Difference / Deviation Map
    axes[3].set_facecolor("#fafafa")
    for i, s in enumerate(layout.strokes):
        axes[3].plot(s[:, 0], s[:, 1], color="#cccccc", lw=3.5, label="Intended path" if i == 0 else None)
    
    c_map = np.where(d_real_to_intended <= 1.0, "#2ca02c",
             np.where(d_real_to_intended <= 2.0, "#ff7f0e", "#d62728"))
    axes[3].scatter(corrected_real[:, 0], corrected_real[:, 1], c=c_map, s=4, alpha=0.9)
    
    axes[3].scatter([], [], c="#2ca02c", s=25, label="Error < 1.0 mm (Match)")
    axes[3].scatter([], [], c="#ff7f0e", s=25, label="Error 1.0 - 2.0 mm (Slight drift)")
    axes[3].scatter([], [], c="#d62728", s=25, label="Error > 2.0 mm (Defect / overshoot)")
    
    stats_text = (
        f"Mean Error: {te['mean_mm']:.2f} mm\n"
        f"Max Error: {te['max_mm']:.2f} mm\n"
        f"1mm Coverage: {cov1*100:.1f}%\n"
        f"Accuracy: {acc['accuracy']*100:.0f}%"
    )
    axes[3].text(0.03, 0.05, stats_text, transform=axes[3].transAxes, fontsize=8.5,
                 verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", alpha=0.9))
    
    axes[3].set_xlim(xmin, xmax)
    axes[3].set_ylim(ymax, ymin)
    axes[3].set_aspect("equal")
    axes[3].set_title("4. Deviation & Diff Map\n(Color-coded error to intended)", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Page X (mm)")
    axes[3].set_ylabel("Page Y (mm)")
    axes[3].legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[3].grid(True, linestyle="--", alpha=0.3)
    
    fig.suptitle(f"Demonstration Comparison: {p['title']}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_demo = out_dir / f"{p['name']}_multi_panel_demo.png"
    fig.savefig(str(out_demo), bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_demo.name}")
    
    # -------------------------------------------------------------
    # FIGURE 2: CLEAN SIDE-BY-SIDE (Zoomed Out, Sim vs Real, No Overlap)
    # -------------------------------------------------------------
    fig_sbs, (ax_sim, ax_real) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    ax_sim.set_facecolor("#ffffff")
    for i, s in enumerate(layout.strokes):
        ax_sim.plot(s[:, 0], s[:, 1], color="#e8e8e8", lw=3.0, linestyle="--")
    for i, s in enumerate(sim_strokes):
        ax_sim.plot(s[:, 0], s[:, 1], color="#1f77b4", lw=2.4, label="Simulation (PyBullet)" if i == 0 else None)
    ax_sim.set_xlim(xmin, xmax)
    ax_sim.set_ylim(ymax, ymin)
    ax_sim.set_aspect("equal")
    ax_sim.set_title("PyBullet Simulation Trajectory", fontsize=12, fontweight="bold")
    ax_sim.set_xlabel("Page X (mm)")
    ax_sim.set_ylabel("Page Y (mm)")
    ax_sim.legend(loc="upper right", fontsize=9)
    ax_sim.grid(True, linestyle="--", alpha=0.3)
    
    ax_real.set_facecolor("#ffffff")
    for i, s in enumerate(layout.strokes):
        ax_real.plot(s[:, 0], s[:, 1], color="#e8e8e8", lw=3.0, linestyle="--")
    ax_real.scatter(corrected_real[:, 0], corrected_real[:, 1], color="#b2182b", s=3.5, alpha=0.85,
                    label=f"Real Ink (Physical Robot, Scale {fit['scale_factor']:.2f}x)")
    ax_real.set_xlim(xmin, xmax)
    ax_real.set_ylim(ymax, ymin)
    ax_real.set_aspect("equal")
    ax_real.set_title("Physical Robot Drawing (Aligned & Zoomed Out)", fontsize=12, fontweight="bold")
    ax_real.set_xlabel("Page X (mm)")
    ax_real.set_ylabel("Page Y (mm)")
    ax_real.legend(loc="upper right", fontsize=9)
    ax_real.grid(True, linestyle="--", alpha=0.3)
    
    fig_sbs.suptitle(f"Clean Side-by-Side Comparison: {p['title']}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_sbs = out_dir / f"{p['name']}_side_by_side.png"
    fig_sbs.savefig(str(out_sbs), bbox_inches="tight")
    plt.close(fig_sbs)
    print(f"Generated {out_sbs.name}")
    
    # -------------------------------------------------------------
    # FIGURE 3: CLEAN DEVIATION & DIFF MAP
    # -------------------------------------------------------------
    fig_diff, ax_diff = plt.subplots(figsize=(9, 6), dpi=150)
    ax_diff.set_facecolor("#ffffff")
    for i, s in enumerate(layout.strokes):
        ax_diff.plot(s[:, 0], s[:, 1], color="#cccccc", lw=4.0, label="Intended Reference" if i == 0 else None)
    
    ax_diff.scatter(corrected_real[:, 0], corrected_real[:, 1], c=c_map, s=5, alpha=0.9)
    ax_diff.scatter([], [], c="#2ca02c", s=30, label="Error < 1.0 mm (Good match)")
    ax_diff.scatter([], [], c="#ff7f0e", s=30, label="Error 1.0 - 2.0 mm (Minor drift)")
    ax_diff.scatter([], [], c="#d62728", s=30, label="Error > 2.0 mm (Defect / large error)")
    
    ax_diff.text(0.03, 0.05, stats_text, transform=ax_diff.transAxes, fontsize=10,
                 verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888888", alpha=0.95))
    
    ax_diff.set_xlim(xmin, xmax)
    ax_diff.set_ylim(ymax, ymin)
    ax_diff.set_aspect("equal")
    ax_diff.set_title(f"Physical Deviation & Diff Map: {p['title']}", fontsize=12, fontweight="bold")
    ax_diff.set_xlabel("Page X (mm)")
    ax_diff.set_ylabel("Page Y (mm)")
    ax_diff.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_diff.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    out_diff = out_dir / f"{p['name']}_diff_map.png"
    fig_diff.savefig(str(out_diff), bbox_inches="tight")
    plt.close(fig_diff)
    print(f"Generated {out_diff.name}")
    
    metadata[p["name"]] = {
        "id": p["id"],
        "name": p["name"],
        "title": p["title"],
        "target": p["target"],
        "policy": p["policy"],
        "photo_file": p["photo"],
        "defect_note": p["defect_note"],
        "scale_factor": round(float(fit["scale_factor"]), 3),
        "rotation_deg": round(float(fit["rotation_deg"]), 2),
        "mean_error_mm": round(float(te["mean_mm"]), 3),
        "max_error_mm": round(float(te["max_mm"]), 3),
        "coverage_1mm": round(float(cov1), 3),
        "accuracy": round(float(acc["accuracy"]), 3),
        "figures": {
            "multi_panel_demo": str(out_demo.relative_to(repo_root)),
            "side_by_side": str(out_sbs.relative_to(repo_root)),
            "diff_map": str(out_diff.relative_to(repo_root)),
            "annotated_lab_photo": str(out_raw.relative_to(repo_root)),
        }
    }

meta_path = out_dir / "demonstrations.json"
meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Saved metadata to {meta_path}")
print("All demonstration comparison figures generated successfully in docs/figures/E11_demonstration/")
