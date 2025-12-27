# 18-Session Plan: Long-Horizon Manipulation SOTA

> **Current Status:** Session 6 of 18 — *Success Detector Training*

**Research Question:**  
> How can we reduce compounding error and increase success on multi-step (≥10 sub-steps) manipulation tasks across embodiments without sacrificing single-step performance?

**Hypotheses:**
- **H1:** Hierarchical subgoal control (predicting K-step visual subgoals and conditioning a low-level policy on them) increases long-horizon success vs. flat RT-X by reducing open-loop drift and re-grasp failures.
- **H2:** Uncertainty-gated replanning (per-step predictive uncertainty → trigger new subgoal) improves recovery rate and reduces time-to-success without hurting single-step success.

---

## Overview

| Phase | Sessions | Focus | Status |
|-------|----------|-------|--------|
| **1. Data Infrastructure** | 1–4 | Make data trainable for long-horizon | ✅ Complete |
| **2. Evaluation Infrastructure** | 5–7 | Fast, automatic, repeatable evaluation | 🔄 In Progress |
| **3. Baseline & Understanding** | 8–10 | Implement baseline, understand SOTA | ⬚ Pending |
| **4. H1 — Hierarchical Subgoals** | 11–14 | Core contribution #1 | ⬚ Pending |
| **5. H2 — Uncertainty Replanning** | 15–17 | Core contribution #2 | ⬚ Pending |
| **6. Final Evaluation & Paper** | 18 | Full eval, paper prep | ⬚ Pending |

---

## Phase 1: Data Infrastructure (Sessions 1–4) ✅

### Session 1: Dataset Catalog & Exploration ✅

**Status:** Complete

**Objectives:**
- Create master catalog of all 12 OXE datasets
- Understand structure variations across datasets

**Pre-Session Reading:**
- [Open X-Embodiment Paper](https://arxiv.org/abs/2310.08864) — Sections 1-3

**Session Activities (1 hr):**
1. Review OXE dataset structure (TFRecord format, feature keys)
2. Discuss variations across datasets (cameras, action spaces, frequencies)
3. Design catalog schema together

**Post-Session Work:**
- Create `catalog.json` with all 12 datasets:

```json
{
  "datasets": [
    {
      "name": "taco_play",
      "version": "0.1.0",
      "split_paths": {
        "train": "gs://x-embodiment-imporvement/oxe_v1_0/taco_play/0.1.0/taco_play-train.tfrecord-*",
        "val": "gs://x-embodiment-imporvement/oxe_v1_0/taco_play/0.1.0/taco_play-val.tfrecord-*"
      },
      "modalities": ["rgb_static", "rgb_gripper", "proprio", "text"],
      "action_dim": 7,
      "action_space": "delta_ee",
      "fps": 5,
      "avg_horizon": null
    }
  ]
}
```

**Deliverable:** `data/catalog.json` with all 12 datasets populated

---

### Session 2: Action Normalization ✅

**Status:** Complete

**Objectives:**
- Compute per-dataset action statistics
- Implement normalization pipeline

**Pre-Session Reading:**
- [RT-1 Paper](https://arxiv.org/abs/2212.06817) — Section 3 (Action Tokenization)

**Session Activities (1 hr):**
1. Review action space differences across datasets
2. Discuss normalization strategies (per-dataset, global, per-action-dim)
3. Design statistics computation pipeline

**Post-Session Work:**
- Script to compute mean/std for each action dimension per dataset
- Save to `data/scales/<dataset>.json`:

```json
{
  "mean": [0.001, 0.002, -0.001, 0.0, 0.0, 0.0, 0.5],
  "std": [0.05, 0.04, 0.03, 0.1, 0.1, 0.1, 0.3],
  "action_names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
}
```

**Deliverable:** `data/scales/*.json` for all 12 datasets + normalization utility functions

---

### Session 3: Sequence Packing ✅

**Status:** Complete

**Objectives:**
- Create windowed sequences for long-horizon training
- Optimize storage format for training efficiency

**Pre-Session Reading:**
- [Octo Paper](https://arxiv.org/abs/2405.12213) — Section 3 (Data Processing)

**Session Activities (1 hr):**
1. Discuss window sizes (16 for short-horizon baseline, 64 for long-horizon)
2. Design stride and overlap strategy
3. Plan GCS output structure

**Post-Session Work:**
- Script to pack sequences into TFRecord/Parquet:

```
gs://x-embodiment-imporvement/sequences/
├── len=16/
│   ├── taco_play/
│   ├── bridge/
│   └── ...
└── len=64/
    ├── taco_play/
    ├── bridge/
    └── ...
```

**Deliverable:** Sequence packing script + GCS upload for all datasets

---

### Session 4: Data Validation & Visualization ✅

**Status:** Complete

**Objectives:**
- Validate packed data quality
- Create diagnostic visualizations

**Pre-Session Reading:**
- Review any data quality papers/blogs (optional)

**Session Activities (1 hr):**
1. Review histograms of action magnitudes across datasets
2. Identify problematic patterns (zero-motion frames, outliers)
3. Sample sequences → GIF for visual inspection

**Post-Session Work:**
- Generate validation report:
  - Histogram: action magnitudes per dataset
  - Table: % zero-motion frames per dataset
  - GIFs: 5 random sequences per dataset
- Fix any data issues discovered

**Deliverable:** `data/validation_report.md` + sample GIFs in `data/samples/`

---

## Phase 2: Evaluation Infrastructure (Sessions 5–7) 🔄

### Session 5: Test Set Lockdown & Manifests ✅

**Status:** Complete

**Objectives:**
- Define held-out evaluation sets
- Create reproducible test manifests

**Pre-Session Reading:**
- [SIMPLER Benchmark Paper](https://arxiv.org/abs/2405.05941) — Evaluation methodology

**Session Activities (1 hr):**
1. Discuss test set selection criteria (scene diversity, task coverage)
2. Define split strategy (% held out, stratification)
3. Design manifest format

**Post-Session Work:**
- Create `eval/test_manifest.json`:

```json
{
  "taco_play": {
    "test_episodes": ["ep_001", "ep_045", "ep_102"],
    "held_out_objects": ["red_mug", "blue_bowl"],
    "num_episodes": 50
  }
}
```

**Deliverable:** `eval/test_manifest.json` with locked test sets for all datasets

---

### Session 6: Success Detector Training 🔄

**Status:** 🔄 IN PROGRESS

**Objectives:**
- Train lightweight success classifiers
- Enable automatic evaluation without human labeling

**Pre-Session Reading:**
- [Goal-Conditioned Imitation Learning](https://arxiv.org/abs/1906.05838) — Success detection sections

**Session Activities (1 hr):**
1. Review success detection approaches (final frame classifier, video classifier)
2. Design labeling strategy (100-500 examples per task family)
3. Choose model architecture (MobileNet, EfficientNet-Lite)

**Post-Session Work:**
- Label success/failure for sample trajectories
- Train classifiers per task family
- Export as ONNX for fast inference

**Deliverable:** `eval/success_detectors/` with ONNX models + accuracy report

**⚠️ Decision Point:** How to acquire labels?
- [ ] Manual labeling (time-intensive but accurate)
- [ ] Heuristic-based (distance to goal < threshold)
- [ ] Existing labels in dataset metadata
- [ ] **Decision needed this session**

---

### Session 7: Metrics Code & Long-Horizon Suites

**Objectives:**
- Implement evaluation metrics
- Define multi-step task chains for long-horizon testing

**Pre-Session Reading:**
- [SuSIE Paper](https://arxiv.org/abs/2310.08864) — Evaluation metrics

**Session Activities (1 hr):**
1. Review metrics: success@K, recovery rate, constraint violations/hour
2. Design long-horizon suites (3-step, 5-step chains)
3. Implement and unit test metrics code

**Post-Session Work:**
- Create `eval/suites/3step.json` and `eval/suites/5step.json`:

```json
{
  "suite_name": "3step_pick_place",
  "tasks": [
    {"step": 1, "action": "reach", "success_detector": "reach_v1.onnx"},
    {"step": 2, "action": "grasp", "success_detector": "grasp_v1.onnx"},
    {"step": 3, "action": "place", "success_detector": "place_v1.onnx"}
  ],
  "max_steps_per_subtask": 50,
  "total_timeout": 300
}
```

- Implement metrics in `eval/metrics.py`:
  - `success_at_k(predictions, labels, k)`
  - `recovery_rate(trajectories)`
  - `constraint_violations_per_hour(trajectories, limits)`
  - `success_vs_horizon_curve(results)`

**Deliverable:** `eval/metrics.py` with unit tests + suite JSON files

---

## Phase 3: Baseline & Understanding (Sessions 8–10)

### Session 8: SOTA Paper Deep Dive

**Objectives:**
- Understand current SOTA approaches
- Identify gaps our approach addresses

**Pre-Session Reading (REQUIRED — discuss in session):**
- [RT-X Paper](https://arxiv.org/abs/2310.08864) — Full paper
- [Octo Paper](https://arxiv.org/abs/2405.12213) — Sections 1-4

**Session Activities (1 hr):**
1. Present and discuss RT-X architecture
2. Present and discuss Octo approach
3. Identify limitations for long-horizon tasks
4. Map to our hypotheses (H1, H2)

**Post-Session Work:**
- Write summary document comparing approaches
- Identify specific architectural decisions to adopt/avoid

**Deliverable:** `docs/sota_analysis.md` comparing RT-X, Octo, and gap analysis

---

### Session 9: Baseline Implementation

**Objectives:**
- Implement or adapt existing baseline model
- Establish performance floor

**Pre-Session Reading:**
- [OpenVLA Paper](https://arxiv.org/abs/2406.09246) — Architecture section

**Session Activities (1 hr):**
1. Decide: train from scratch vs. fine-tune existing checkpoint
2. Design training pipeline (data loading, model, optimizer)
3. Set up GCP training infrastructure

**Post-Session Work:**
- Implement baseline training script
- Launch training on GCP
- Log to Weights & Biases / TensorBoard

**⚠️ Decision Point:** Baseline model choice
- [ ] Train RT-1 style from scratch (more control, more compute)
- [ ] Fine-tune Octo checkpoint (faster, proven)
- [ ] Fine-tune OpenVLA (best performance, most compute)
- [ ] **TBD: Decide based on compute budget**

**Deliverable:** Training script + initial training run launched

---

### Session 10: Baseline Evaluation & Analysis

**Objectives:**
- Evaluate baseline on our test sets
- Identify failure modes for long-horizon tasks

**Pre-Session Reading:**
- Review failure analysis methodologies

**Session Activities (1 hr):**
1. Review baseline metrics on single-step tasks
2. Analyze long-horizon performance degradation
3. Categorize failure modes (drift, re-grasp, accumulation)

**Post-Session Work:**
- Run full evaluation suite on baseline
- Generate failure mode analysis:
  - Success vs. horizon curve
  - Common failure patterns (with example GIFs)
  - Per-dataset breakdown

**Deliverable:** `results/baseline/` with metrics + `docs/baseline_analysis.md`

---

## Phase 4: H1 — Hierarchical Subgoal Control (Sessions 11–14)

### Session 11: Subgoal Prediction Design

**Objectives:**
- Design subgoal representation
- Plan subgoal predictor architecture

**Pre-Session Reading:**
- [SuSIE: Subgoal Synthesis via Image Editing](https://arxiv.org/abs/2310.12931) — Core approach
- [Hierarchical Decision Transformer](https://arxiv.org/abs/2209.10447) — Alternative perspective

**Session Activities (1 hr):**
1. Discuss subgoal representations (image, latent, waypoint)
2. Review SuSIE's image-editing approach
3. Design our subgoal predictor (predict image K steps ahead)
4. Decide K (subgoal horizon): 8? 16? variable?

**Post-Session Work:**
- Implement subgoal predictor architecture
- Set up training data pipeline (pairs: current → K-step-ahead images)

**Deliverable:** Subgoal predictor model code + data pipeline

---

### Session 12: Subgoal Predictor Training

**Objectives:**
- Train subgoal prediction model
- Validate subgoal quality

**Pre-Session Reading:**
- [Diffusion Models for Robotics](https://arxiv.org/abs/2303.04137) — If using diffusion

**Session Activities (1 hr):**
1. Review training progress
2. Discuss subgoal quality metrics (FID, LPIPS, human eval)
3. Analyze failure cases in subgoal prediction

**Post-Session Work:**
- Complete subgoal predictor training
- Evaluate subgoal quality:
  - Visual inspection (predicted vs. actual K-step-ahead)
  - Quantitative metrics
- Iterate on architecture if needed

**Deliverable:** Trained subgoal predictor + quality evaluation report

---

### Session 13: Subgoal-Conditioned Low-Level Policy

**Objectives:**
- Modify policy to condition on subgoal images
- Integrate subgoal predictor with policy

**Pre-Session Reading:**
- [Goal-Conditioned Behavior Cloning](https://arxiv.org/abs/2004.00567) — Conditioning mechanisms

**Session Activities (1 hr):**
1. Design conditioning mechanism (concatenation, cross-attention, FiLM)
2. Discuss training strategy (joint vs. separate)
3. Plan integration architecture

**Post-Session Work:**
- Implement subgoal-conditioned policy
- Train on subgoal-conditioned objective
- Compare to baseline on single-step metrics (ensure no regression)

**Deliverable:** Subgoal-conditioned policy + single-step comparison report

---

### Session 14: Hierarchical Integration & H1 Evaluation

**Objectives:**
- Integrate subgoal predictor + low-level policy
- Evaluate H1 on long-horizon tasks

**Pre-Session Reading:**
- Review hierarchical RL integration patterns

**Session Activities (1 hr):**
1. Review integration (predict subgoal every K steps, execute low-level for K steps)
2. Discuss failure cases and debugging strategies
3. Plan full evaluation

**Post-Session Work:**
- Run H1 system on full evaluation suite
- Compare to baseline:
  - Single-step success (should be similar)
  - Long-horizon success (should improve)
  - Success vs. horizon curve
- Document improvement and failure modes

**Deliverable:** `results/h1/` with full metrics + `docs/h1_analysis.md`

---

## Phase 5: H2 — Uncertainty-Gated Replanning (Sessions 15–17)

### Session 15: Uncertainty Estimation

**Objectives:**
- Add uncertainty estimation to policy
- Calibrate uncertainty thresholds

**Pre-Session Reading:**
- [Ensemble Uncertainty in Deep Learning](https://arxiv.org/abs/1612.01474)
- [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142)

**Session Activities (1 hr):**
1. Discuss uncertainty estimation methods (ensemble, MC dropout, learned)
2. Choose approach based on compute constraints
3. Design calibration methodology

**Post-Session Work:**
- Implement uncertainty estimation in policy
- Collect uncertainty statistics on validation set
- Calibrate threshold: what uncertainty level → trigger replan?

**Deliverable:** Uncertainty-aware policy + calibration report

---

### Session 16: Replanning Logic

**Objectives:**
- Implement uncertainty-triggered replanning
- Integrate with H1 hierarchical system

**Pre-Session Reading:**
- [Model Predictive Control in Robotics](https://arxiv.org/abs/2012.09500) — Replanning concepts

**Session Activities (1 hr):**
1. Design replanning trigger logic
2. Discuss replan frequency limits (avoid oscillation)
3. Integrate with H1 subgoal system

**Post-Session Work:**
- Implement replanning controller:
  - Monitor per-step uncertainty
  - If uncertainty > threshold → request new subgoal
  - Rate-limit replanning to avoid thrashing
- Integration testing on sample trajectories

**Deliverable:** Replanning controller code + integration tests

---

### Session 17: H2 Evaluation & Combined System

**Objectives:**
- Evaluate H1+H2 combined system
- Compare against baselines

**Pre-Session Reading:**
- None (focus on evaluation)

**Session Activities (1 hr):**
1. Review H2 metrics (recovery rate, time-to-success)
2. Analyze when replanning helps vs. hurts
3. Compare: Baseline vs. H1 vs. H1+H2

**Post-Session Work:**
- Full evaluation of combined H1+H2 system:
  - Success@K
  - Recovery rate (how often does replanning save a failing trajectory?)
  - Time-to-success (does replanning add latency?)
  - Success vs. horizon curve
- Per-dataset and per-task breakdown

**Deliverable:** `results/h1_h2/` with full metrics + `docs/h2_analysis.md`

---

## Phase 6: Final Evaluation & Paper (Session 18)

### Session 18: Final Evaluation & Paper Preparation

**Objectives:**
- Complete final evaluation across all benchmarks
- Prepare paper draft outline

**Pre-Session Reading:**
- Review top CoRL/ICRA paper structures

**Session Activities (1 hr):**
1. Review final results across all metrics
2. Discuss ablations needed (H1 only, H2 only, H1+H2)
3. Outline paper structure and key claims
4. Assign writing responsibilities

**Post-Session Work:**
- Final ablation studies if needed
- Begin paper draft:
  - Abstract
  - Introduction (RQ, contributions)
  - Method (H1, H2)
  - Experiments (baselines, metrics, results)
  - Analysis (failure modes, ablations)
  - Conclusion

**Deliverables:**
- `results/final/` with all metrics and comparisons
- `paper/outline.md` with detailed paper structure
- `paper/figures/` with key visualizations

**⚠️ Decision Point:** Paper venue
- [ ] CoRL 2025 (robotics-focused)
- [ ] ICRA 2026 (broader robotics)
- [ ] NeurIPS 2025 (ML audience)
- [ ] **TBD: Decide based on results timeline**

---

## Summary: Deliverables by Session

| Session | Key Deliverable | Status |
|---------|----------------|--------|
| 1 | `data/catalog.json` | ✅ |
| 2 | `data/scales/*.json` + normalization utils | ✅ |
| 3 | Sequence packing script + GCS upload | ✅ |
| 4 | `data/validation_report.md` + sample GIFs | ✅ |
| 5 | `eval/test_manifest.json` | ✅ |
| **6** | **`eval/success_detectors/*.onnx`** | **🔄** |
| 7 | `eval/metrics.py` + suite JSONs | ⬚ |
| 8 | `docs/sota_analysis.md` | ⬚ |
| 9 | Training script + initial run | ⬚ |
| 10 | `results/baseline/` + analysis doc | ⬚ |
| 11 | Subgoal predictor architecture | ⬚ |
| 12 | Trained subgoal predictor | ⬚ |
| 13 | Subgoal-conditioned policy | ⬚ |
| 14 | `results/h1/` + H1 analysis | ⬚ |
| 15 | Uncertainty-aware policy | ⬚ |
| 16 | Replanning controller | ⬚ |
| 17 | `results/h1_h2/` + H2 analysis | ⬚ |
| 18 | `results/final/` + paper outline | ⬚ |

---

## Open Decisions (TBD)

| Decision | Options | Decide By | Status |
|----------|---------|-----------|--------|
| Success labeling strategy | Manual / Heuristic / Metadata | Session 6 | 🔄 Deciding now |
| Baseline model | Train from scratch / Fine-tune Octo / Fine-tune OpenVLA | Session 8 | ⬚ Pending |
| Subgoal horizon K | 8 / 16 / variable | Session 11 | ⬚ Pending |
| Uncertainty method | Ensemble / MC Dropout / Learned | Session 15 | ⬚ Pending |
| Paper venue | CoRL / ICRA / NeurIPS | Session 18 | ⬚ Pending |
| Demo format | Simulation video / Real robot (if available) | Session 17 | ⬚ Pending |

---

## Recommended Paper Reading Schedule

| Session | Paper(s) |
|---------|----------|
| 1 | Open X-Embodiment (Sections 1-3) |
| 2 | RT-1 (Section 3: Action Tokenization) |
| 3 | Octo (Section 3: Data Processing) |
| 5 | SIMPLER Benchmark |
| 6 | Goal-Conditioned IL |
| 8 | **RT-X (Full)** + **Octo (Sections 1-4)** |
| 9 | OpenVLA |
| 11 | **SuSIE** + Hierarchical DT |
| 15 | Ensemble Uncertainty + MC Dropout |
| 16 | MPC in Robotics |

---

## Success Criteria

**Minimum (Paper Submission):**
- [ ] H1+H2 system outperforms baseline on 3-step and 5-step suites
- [ ] No regression on single-step performance (≤2% drop)
- [ ] Ablation showing contribution of H1 and H2 separately
- [ ] Clear success vs. horizon curve showing improvement

**Stretch (SOTA + Demo):**
- [ ] Beat published Octo/RT-X numbers on OXE benchmark
- [ ] Working demo video on at least 3 datasets
- [ ] Generalization to held-out embodiment
- [ ] Open-source code release

---

## File Structure (End State)

```
project/
├── data/
│   ├── catalog.json
│   ├── scales/
│   │   ├── taco_play.json
│   │   ├── bridge.json
│   │   └── ...
│   ├── validation_report.md
│   └── samples/
│       └── *.gif
├── eval/
│   ├── test_manifest.json
│   ├── success_detectors/
│   │   └── *.onnx
│   ├── suites/
│   │   ├── 3step.json
│   │   └── 5step.json
│   └── metrics.py
├── models/
│   ├── baseline/
│   ├── subgoal_predictor/
│   └── h1_h2_policy/
├── results/
│   ├── baseline/
│   ├── h1/
│   ├── h1_h2/
│   └── final/
├── docs/
│   ├── 18_session_plan.md
│   ├── sota_analysis.md
│   ├── baseline_analysis.md
│   ├── h1_analysis.md
│   └── h2_analysis.md
└── paper/
    ├── outline.md
    └── figures/
```

