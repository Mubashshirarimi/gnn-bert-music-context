# Results summary

Auto-generated from `results/metrics.json` by `scripts/make_report.py`.

## Multi-label tagging (MTAT test split, top-50 tags)

| Model | Macro-F1 | Micro-F1 | AUC-PR | ROC-AUC |
|---|---|---|---|---|
| B1: prior-frequency predictor | 0.0000 | 0.0000 | 0.0670 | 0.4971 |
| B4: PCA + MLP (hand-crafted) | 0.3876 | 0.4604 | 0.3722 | 0.8712 |
| B2: CNN on mel-spectrogram | 0.4383 | 0.5081 | 0.4329 | 0.9079 |
| Task 1: BERT (metadata only) | 0.2983 | 0.3271 | 0.2741 | 0.8136 |
| Task 1: BERT (metadata + aux tags) | 0.3322 | 0.3991 | 0.3339 | 0.8461 |
| Task 2: GraphSAGE | 0.4015 | 0.4646 | 0.3870 | 0.8848 |
| Task 2: GAT | 0.4023 | 0.4590 | 0.3773 | 0.8850 |
| Task 3 ablation: BERT only | 0.3302 | 0.3955 | 0.3259 | 0.8344 |
| Task 3 ablation: GNN only | 0.3981 | 0.4278 | 0.3840 | 0.8833 |
| Task 3 ablation: early concat | 0.4061 | 0.4771 | 0.3977 | 0.8837 |
| Task 3: GNN-BERT cross-attention | 0.3853 | 0.4448 | 0.3732 | 0.8798 |

## Fusion ablation (Task 3)

| Fusion variant | Macro-F1 | AUC-PR | Graph coherence |
|---|---|---|---|
| BERT only | 0.3302 | 0.3259 | -- |
| GNN only | 0.3981 | 0.3840 | 0.9988 |
| early concat | 0.4061 | 0.3977 | 0.9992 |
| GNN-BERT cross-attention | 0.3853 | 0.3732 | 1.0000 |

## Cross-modal retrieval (Task 4)

| Direction | R@1 | R@5 | R@10 | Median rank |
|---|---|---|---|---|
| Caption -> Audio | 0.0276 | 0.0815 | 0.1236 | 131.0 |
| Audio -> Caption | 0.0214 | 0.0478 | 0.0706 | 279.0 |
