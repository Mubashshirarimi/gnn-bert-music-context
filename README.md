# GNN-BERT for Understanding Context from Music

Supervised neural-network project for **CSE425 / EEE474 / CSE715**.

A hybrid **BERT + Graph Neural Network** system that predicts musical context
from audio structure and text. Rather than modelling a track as a flat
spectrogram, each clip becomes a **music structure graph** whose nodes are
time segments and template-matched chords, and whose edges encode temporal
adjacency, acoustic repetition, chord transitions and chord membership. A BERT
encoder reads the clip's natural-language context, and a cross-attention layer
fuses the two.

All four tasks from the project brief are implemented:

| Task | Model | Entry point |
|------|-------|-------------|
| 1 (Easy) | BERT multi-label tag classifier | `python -m src.train --task 1` |
| 2 (Medium) | GraphSAGE / GAT on structure graphs | `python -m src.train --task 2` |
| 3 (Hard) | GNN-BERT cross-attention fusion | `python -m src.train --task 3` |
| 4 (Advanced) | Contrastive dual encoder + retrieval | `python -m src.train --task 4` |

---

## 1. Quick start

```bash
# 1. Environment (Python 3.10+; this repo was developed on 3.14)
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Verify the whole stack on synthetic audio - no dataset needed (~1 min)
python -m tests.test_smoke

# 3. Get the data (~3 GB) and build graphs
python scripts/download_data.py
python scripts/preprocess.py

# 4. Train and evaluate everything
python scripts/run_all.py
```

`scripts/run_all.py` runs every task, both required baselines, the ablation
sweep and the evaluation pass, writing `results/metrics.json` and
`results/plots/`.

---

## 2. Dataset

**MagnaTagATune** — 25,860 clips of ~29 s, annotated with 188 free-text tags.

| Split | Clips | Source |
|-------|-------|--------|
| train | 18,706 | folders `0`–`b` |
| val | 1,825 | folder `c` |
| test | 5,329 | folders `d`–`f` |

Splits follow the standard Won et al. partition shipped as `*_gt_mtt.tsv`.
Because MTAT's directories group clips by **album folder**, this split is also
what prevents artist and album leakage across train/test — clips from one album
never straddle the boundary.

**Labels.** The 188 raw tags are collapsed with the standard synonym merge
(`female vocal`/`female vocals`/`woman singing` → `female`, and 26 other
groups), then reduced to the 50 most frequent. `scripts/verify_labels.py`
cross-checks the rebuilt matrix against the shipped ground-truth vectors.

### The text modality: an honest note

MagnaTagATune ships **no lyrics and no captions**. Its only text *is* the tag
set — which is the prediction target. Feeding tags to BERT to predict tags would
be circular and would produce a meaninglessly high Task 1 score.

The text modality is therefore built from two sources that are **disjoint from
the labels**:

1. **Catalogue metadata** — track title, artist and album from
   `clip_info_final.csv`, rendered as a sentence:
   `"The track is titled 'BWV54 - I Aria' by American Bach Soloists, from the album 'J.S. Bach Solo Cantatas'."`
2. **Long-tail annotations** — the ~130 tags that fall *outside* the top-50
   label set, appended as `"Listeners also described it as: ..."`.

Set `text.use_auxiliary_tags: false` in `config.yaml` (or
`--set text.use_auxiliary_tags=false`) to train on metadata alone. Both settings
are reported, because they differ materially: the auxiliary tags are worth
+0.034 Macro-F1 and +0.060 AUC-PR, so it matters which one a number came from.
Only 21.5 % of clips carry any auxiliary tag, which is why the gap is not larger
— and why cross-attention has so little text to work with (see Results).

Context strings are rebuilt from `manifest.csv` at load time rather than read
from the preprocessed graphs, so text-side config changes take effect without
re-running the audio pipeline.

Note that the folder split makes this genuinely hard: artists in the test split
never appear in training, so BERT cannot memorise artist→genre associations.
A modest Task 1 score is the expected and correct outcome, and it is exactly
what makes the Task 3 fusion gain meaningful.

---

## 3. Method

### 3.1 Audio features

22,050 Hz mono → 128-bin log-mel spectrogram and 12-bin chromagram
(`n_fft=2048`, `hop=512`). Chroma uses a Gaussian pitch-class filterbank
reimplemented in `src/audio_features.py`, so the project depends only on
`torch`/`torchaudio` — no `librosa`, and therefore no `numba`, which does not
yet support Python 3.14.

Clips are cut into 5 s windows at 50 % overlap (≤16 per clip), and each window
is mean/std pooled into a 280-dim node feature.

### 3.2 Graph construction

One flat `Data` object per track, with node and edge types recorded as
attributes so stock `SAGEConv`/`GATConv` layers work unchanged:

| Edge type | From → To | Meaning |
|-----------|-----------|---------|
| 0 | segment ↔ segment | temporal adjacency |
| 1 | segment ↔ segment | cosine similarity > 0.85 (repetition / song form) |
| 2 | chord → chord | observed transition, weighted by count |
| 3 | segment ↔ chord | which triads sound during which window |

Chords come from template matching 24 major/minor triads against chroma,
median-filtered and run-length collapsed.

### 3.3 Models

**Task 1** — BERT-base, last 4 layers unfrozen, masked mean pooling, linear head,
`BCEWithLogitsLoss` with per-tag `pos_weight`.

**Task 2** — 3-layer GraphSAGE with residual connections and batch norm,
mean+max readout:

```
h_i^(l+1) = σ( W^(l) · CONCAT( h_i^(l), MEAN_{j∈N(i)} h_j^(l) ) )
g         = CONCAT( mean_i h_i^(L), max_i h_i^(L) )
```

**Task 3** — the graph embedding forms a single query attending over BERT's
token states:

```
A = softmax(QKᵀ/√d),  Q = gW_Q,  K = H_text W_K
z = CONCAT(g, A·H_text)
```

Four fusion modes share one interface (`bert_only`, `gnn_only`, `concat`,
`cross_attention`), so the ablation is a true apples-to-apples comparison — same
heads, same optimiser, same schedule.

**Task 4** — dual encoder projecting graphs and text into a shared 256-d space,
trained with symmetric InfoNCE and a learnable temperature. Clips cut from the
same source track share a metadata string, so they are masked out of the
negatives instead of being scored as wrong answers.

### 3.4 Class imbalance and thresholds

Even among the top 50 tags, positives are roughly 1–10 % of clips. Two measures
address this:

* `pos_weight = #neg/#pos` per tag, clipped at 20 to keep training stable.
* Per-tag decision thresholds fitted **on validation only**, then frozen for
  test. Fitting thresholds on test would inflate F1 substantially; the code
  keeps the two strictly separate.

---

## 4. Results

Numbers are produced by `python scripts/run_all.py` and written to
`results/metrics.json`. See `report/final_report.pdf` for the discussion.

Measured on the MTAT test split (4,392 clips, top-50 tags). Thresholds fitted on
validation, frozen for test. Single seed.

| Model | Macro-F1 | Micro-F1 | AUC-PR | ROC-AUC |
|-------|----------|----------|--------|---------|
| B1 prior-frequency | 0.0000 | 0.0000 | 0.0670 | 0.4971 |
| Task 1 BERT (metadata only) | 0.2983 | 0.3271 | 0.2741 | 0.8136 |
| Task 1 BERT (metadata + aux tags) | 0.3322 | 0.3991 | 0.3339 | 0.8461 |
| B4 PCA + MLP | 0.3876 | 0.4604 | 0.3722 | 0.8712 |
| Task 2 GraphSAGE | 0.4015 | 0.4646 | 0.3870 | 0.8848 |
| Task 2 GAT | 0.4023 | 0.4590 | 0.3773 | 0.8850 |
| Task 3 GNN-BERT cross-attention | 0.3853 | 0.4448 | 0.3732 | 0.8798 |
| **Task 3 GNN-BERT concat** | **0.4061** | 0.4771 | **0.3977** | 0.8837 |
| B2 CNN on mel | **0.4383** | **0.5081** | **0.4329** | **0.9079** |

| Retrieval (Task 4) | R@1 | R@5 | R@10 | median rank |
|--------------------|-----|-----|------|-------------|
| caption → audio | 0.0276 | 0.0815 | 0.1236 | 131 |
| audio → caption | 0.0214 | 0.0478 | 0.0706 | 279 |

### What the numbers say

* **The setup is sound.** The mel CNN's ROC-AUC of 0.908 matches published
  CNN taggers on this benchmark, which validates the split, labels and metrics.
* **The graph carries real signal** — GraphSAGE (0.4015) beats hand-crafted
  features (0.3876) and text alone (0.3322) — **but does not beat a CNN on the
  raw spectrogram** (0.4383). Pooling 5 s windows into mean/std vectors trades
  local time–frequency detail for relational structure, and on MTAT's largely
  timbral tag vocabulary that trade does not pay off.
* **Fusion helps, and plain concatenation beats cross-attention.** Within the
  controlled ablation, concat lifts the graph branch from 0.3981 → 0.4061
  Macro-F1; cross-attention *drops* it to 0.3853. Context strings average ~17
  words of metadata and only 21.5 % of clips carry any auxiliary tag, so there
  is very little for attention to select between — and extra parameters to
  overfit.
* **The gain is concentrated in the long tail.** Fusion's largest per-tag
  improvements are all rare tags: sitar 0.369→0.516, cello 0.080→0.261, jazz
  0.281→0.416. Text rescues tags the audio branch cannot learn from few
  examples, rather than adding information across the board.
* **Graph coherence ≈ 1.0 indicates over-smoothing.** Three residual message-
  passing layers on 15–25 node graphs drive nearly every edge's endpoints to
  near-identical embeddings. Reported as measured; the obvious fixes are fewer
  layers, dropping the residual path, or sparser similarity edges.

Plots land in `results/plots/`: training curves, per-tag F1/AUC-PR, t-SNE of the
fused representation coloured by genre and mood, and a cross-model comparison
bar chart. Ten qualitative retrieval examples are written to
`results/retrieval_examples/`.

---

## 5. Repository layout

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml               # every hyperparameter; override with --set a.b=c
├── data/
│   ├── raw/mtat/             # downloaded archives + annotations
│   ├── processed/            # graphs_{split}.pt, mel memmaps, labels, vocabulary
│   │   └── examples/         # 20 exported example graphs (submission requirement)
│   └── splits/
├── notebooks/
│   ├── eda.ipynb             # dataset + graph statistics
│   └── demo_context.ipynb    # end-to-end inference on one clip
├── src/
│   ├── audio_features.py     # mel, chroma, segmentation, chord estimation
│   ├── graph_builder.py      # segment + chord graphs
│   ├── bert_encoder.py       # text encoder, Task 1 classifier, context strings
│   ├── gnn_model.py          # GraphSAGE / GAT, mel-CNN baseline
│   ├── fusion_model.py       # cross-attention GNN-BERT, multi-task loss
│   ├── contrastive.py        # Task 4 InfoNCE dual encoder + retrieval
│   ├── dataset.py            # MTAT loading, vocabulary, splits, datasets
│   ├── baselines.py          # PCA + MLP baseline (B4)
│   ├── metrics.py            # F1, AUC-PR, MAE/R², graph coherence
│   ├── train.py              # unified trainer for all tasks
│   ├── evaluate.py           # metrics, plots, t-SNE, retrieval tables
│   └── utils.py              # config, seeding, logging, checkpoints
├── scripts/
│   ├── download_data.py      # fetch MagnaTagATune
│   ├── preprocess.py         # audio → graphs + mel memmaps
│   ├── verify_labels.py      # cross-check vocabulary vs shipped ground truth
│   └── run_all.py            # full pipeline
├── tests/test_smoke.py       # synthetic end-to-end check, no data required
├── results/
│   ├── metrics.json
│   ├── plots/
│   └── retrieval_examples/
└── report/final_report.pdf
```

---

## 6. Reproducibility

* Every hyperparameter lives in `config.yaml`; nothing is hard-coded in the
  training scripts. Override without editing the file:
  `python -m src.train --task 3 --set model.fusion.mode=concat train.epochs=10`
* `src/utils.set_seed` seeds Python, NumPy and torch. Pass
  `deterministic=True` for bit-reproducible GPU runs at some cost in speed.
* Checkpoints store the full config used to produce them, so any result can be
  traced back to its exact settings.
* Hardware used: RTX 3070 Laptop (8 GB), 16 GB RAM. Mixed precision is on by
  default (`train.amp`); with 8 GB, Task 3 at `batch_size: 32` is comfortable.
  Graph tasks use `num_workers: 0` deliberately — Windows spawns worker
  processes that would each copy the in-memory graph store.

## 7. Known limitations

* MTAT has no lyrics or captions, so the "language" side is metadata plus
  long-tail tags rather than rich description. Section 2 explains the choice.
* Emotion regression (valence/arousal) is implemented in `MultiTaskLoss` and
  `metrics.emotion_metrics` but is **disabled by default**: it needs DEAM,
  which is a separate corpus with no clip-level correspondence to MTAT. Enable
  with `model.emotion.enabled: true` after supplying aligned targets.
* Chord estimation is template matching, not a trained chord recogniser. It is
  adequate for building transition graphs but is not a state-of-the-art
  automatic chord estimator.

## 8. Building the report

`report/final_report.pdf` is committed and already contains the measured
numbers. To rebuild after a new run:

```bash
python scripts/make_report.py     # regenerates report/results_tables.tex from metrics.json
cd report && pdflatex final_report && pdflatex final_report
```

The document compiles standalone with the `article` class. For the official
NeurIPS look, drop `neurips_2024.sty` beside it (the preamble picks it up
automatically) or paste the body into the
[NeurIPS 2024 Overleaf template](https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh).
Result tables are generated from `results/metrics.json`, so the report can never
drift from the metrics actually measured.

## 9. Author

**Mubashshira** — ID 22201426, Section 02
Department of Computer Science and Engineering, BRAC University
<mubashshira@g.bracu.ac.bd>

Course: Neural Networks (CSE425 / EEE474 / CSE715)
