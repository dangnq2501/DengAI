"""Build the DengAI score-journey PowerPoint from artifacts/slide_figures."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "slide_figures"
OUT = ROOT / "artifacts" / "DengAI_score_journey.pptx"


def _add_title(slide, title: str, subtitle: str = "") -> None:
    slide.shapes.title.text = title
    if subtitle and len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle


def _add_bullets(slide, lines: list[str], left=0.6, top=1.6, width=8.8, height=5.0) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for index, line in enumerate(lines):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(20 if index == 0 else 18)


def _add_picture(slide, name: str, left=0.5, top=1.5, width=9.0) -> None:
    path = FIG / name
    if not path.is_file():
        return
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))


def _notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # 1 — Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    _add_title(
        slide,
        "DengAI: Predicting Weekly Dengue from Climate",
        "From baseline (~26.7) to multiscale MLP (16.6 hidden-test MAE)\n"
        "MBZUAI · feature_lag_mlp + tree ensemble pipeline",
    )
    _notes(
        slide,
        "Competition: San Juan (936 train / 260 test weeks) and Iquitos (520 / 156). "
        "Metric: mean absolute error on integer case counts. No case lags at test time.",
    )

    # 2 — Objective
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Problem and objective")
    _add_bullets(
        slide,
        [
            "Predict total_cases each week from weather and NDVI (test weather is known).",
            "Separate dynamics: SJ large outbreaks vs IQ smaller, zero-heavy counts.",
            "Cannot use true past case labels across the multi-year test horizon.",
            "Goal: causal climate history φ(t) → small predictor → weekly cases.",
        ],
    )
    _add_picture(slide, "01_target_scale_and_outbreaks.png", top=1.45, width=4.3)
    _notes(slide, "Figure 01: scale and outbreak asymmetry motivate city-specific models.")

    # 3 — Data difficulty
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "SJ has rare peaks; IQ is lower and sparser")
    _add_picture(slide, "01_target_scale_and_outbreaks.png", top=1.35, width=9.2)
    _notes(
        slide,
        "Smoothing or mean predictors hurt SJ MAE. Skewed counts favor models that "
        "preserve amplitude on outbreaks.",
    )

    # 4 — Seasonality
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Seasonality helps but does not determine cases")
    _add_picture(slide, "02_seasonal_case_pattern.png", top=1.35, width=9.2)
    _notes(slide, "Calendar phase plus recent climate both required.")

    # 5 — Trees 22.5
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Tree ensemble baseline — 22.5 MAE (confirmed submission)")
    _add_bullets(
        slide,
        [
            "Features: current climate + Fourier season + lags 1/2/4/8/12/16",
            "+ rolling means 2/4/8/12/16 + NDVI summary + interactions",
            "SJ: Extra Trees (Poisson split) · IQ: Random Forest on log1p(cases)",
            "Confirmed file: artifacts/submission_tree_ensemble.csv",
            "Lesson: delayed climate and nonlinear thresholds matter.",
        ],
        top=1.5,
        width=9.0,
    )
    _notes(slide, "First strong repo baseline; see docs/tree_preprocessing_explained.md.")

    # 6 — Lag windows
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Feature-specific memory (lag windows)")
    _add_picture(slide, "03_temporal_memory_patterns.png", top=1.35, width=9.2)
    _add_bullets(
        slide,
        [
            "Each climate variable keeps a contiguous history block (e.g. 40 weeks).",
            "461 SJ / 380 IQ inputs → three-layer SELU MLP (vendor V10 lineage).",
            "Hidden test ~19.3 → dropout/tuning → 18.8 (seed 42).",
        ],
        top=5.9,
        width=9.0,
    )
    _notes(slide, "DT window search in docs/lag_window_selection_explained.md.")

    # 7 — Raw MLP limits
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Tuned raw-history MLP — 18.8 MAE")
    _add_bullets(
        slide,
        [
            "Architecture: SJ 100→25, IQ 70→18; shared-SJ normalization.",
            "Temporal tuning: weighted CV 15.862 vs 19.1 control (local).",
            "Three-seed ensemble: 19.5 hidden · single seed 42 beats ensemble.",
        ],
        top=1.45,
        width=4.2,
    )
    _add_picture(slide, "04_representation_ablation.png", left=4.6, top=1.35, width=5.0)
    _notes(slide, "Raw+summary ablation failed → redundancy in adjacent weeks.")

    # 8 — Multiscale
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Multiscale summaries — 180 inputs")
    _add_bullets(
        slide,
        [
            "Per weather variable (×16): level, means 2/4/8/13/26/52,",
            "  std 4/13, short−medium contrast, 13-week trend.",
            "+ 4 Fourier season features → 16×11 + 4 = 180.",
            "Same tuned MLP; local weighted MAE ~13.9 vs ~15.9 raw-lag.",
            "SJ-only multiscale hybrid: 16.6 hidden-test MAE (documented).",
        ],
        top=1.45,
        width=4.3,
    )
    _add_picture(slide, "05_sj_prediction_profile.png", left=4.8, top=1.3, width=4.8)
    _notes(slide, "docs/multiscale_16_6_model_explained.md")

    # 9 — Journey chart
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Hidden-test MAE journey (lower is better)")
    _add_picture(slide, "06_leaderboard_journey.png", top=1.4, width=8.5)
    _add_bullets(
        slide,
        [
            "26.7 → 22.5 trees → ~19 raw MLP → 18.8 tuned → 16.6 multiscale",
            "48→12 narrow net: 18.1 — rejected (local win, hidden loss)",
        ],
        top=5.85,
        width=9.0,
    )
    _notes(slide, "Largest gain from representation φ(t), not deeper networks.")

    # 10 — Validation trap
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Local validation ≠ hidden leaderboard")
    _add_picture(slide, "07_local_validation_trap.png", top=1.35, width=9.0)
    _notes(
        slide,
        "Small CV gains after many tries can be noise; outbreak MAE and amplitude "
        "matter for SJ.",
    )

    # 11 — Repository map
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "What lives in this repository")
    _add_bullets(
        slide,
        [
            "src/train_tree_ensemble.py — 22.5 tree pipeline",
            "src/feature_lag_mlp/ — lag MLP, tuning, multiscale representations",
            "src/feature_lag_mlp/representation_train.py — 16.6 candidates",
            "leaderboard_notebook/ — chronological dev blends (separate track)",
            "research/ — reproduction scripts, PCA/ensemble experiments (session work)",
            "docs/slide_flow_and_figures.md + artifacts/slide_figures/*.png",
        ],
        top=1.5,
    )
    _notes(slide, "Canonical package: src/feature_lag_mlp; vendors/ not required.")

    # 12 — Repro & recent work
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Reproducing and extending (2026 session)")
    _add_bullets(
        slide,
        [
            "scripts/reproduce_multiscale_best.py → multiscale + sj_only CSVs",
            "scripts/reproduce_feature_lag_mlp_tuned.py → tuned 3-seed ensemble",
            "research/select_and_export_best_submission.py → holdout + nearest rounding",
            "Best local export: submission_feature_lag_mlp_best_candidate.csv",
            "Note: NumPy fallback when TensorFlow/disk limited; re-run with TF for Keras parity",
            "PCA on multiscale did not beat 180-dim baseline on temporal CV",
        ],
        top=1.45,
    )
    _notes(
        slide,
        "Hidden 16.6 requires Keras training per project docs. Local holdout favored "
        "multiscale seed 42 with nearest-integer rounding over plain truncate.",
    )

    # 13 — Takeaways
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Takeaways")
    _add_bullets(
        slide,
        [
            "1. Represent climate history at multiple timescales, not 400 redundant lags.",
            "2. Keep the predictor small; tune width/dropout per city.",
            "3. Validate in time; report outbreak error, not only mean MAE.",
            "4. Ship city hybrids when only one city improves (SJ multiscale, IQ baseline).",
            "5. Submission to beat for leaderboard story: multiscale_summaries_seed42_sj_only (16.6).",
        ],
        top=1.55,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
