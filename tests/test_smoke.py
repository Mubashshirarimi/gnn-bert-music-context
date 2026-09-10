"""End-to-end smoke tests on synthetic audio - no MagnaTagATune download needed.

    python -m tests.test_smoke          # from the project root
    pytest tests/test_smoke.py -v

Every component is exercised on generated waveforms, so a green run proves the
feature extraction, graph construction, all four models, the loss functions and
the metrics are wired together correctly before any long training job starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio_features import FeatureExtractor, estimate_chords, pool_segments, segment_features
from src.bert_encoder import build_context_string
from src.contrastive import build_duplicate_mask, info_nce_loss, retrieval_metrics
from src.fusion_model import GNNBertFusion, MultiTaskLoss
from src.gnn_model import GNNTagClassifier, MelCNNBaseline
from src.graph_builder import GraphConfig, build_track_graph, graph_summary
from src.metrics import graph_coherence_score, search_thresholds, tagging_metrics

SR = 22050
AUDIO_CFG = {
    "sample_rate": SR, "n_fft": 2048, "hop_length": 512, "n_mels": 128, "n_chroma": 12,
    "fmin": 30.0, "fmax": 11025.0, "segment_seconds": 5.0, "segment_overlap": 0.5, "max_segments": 16,
}
NUM_TAGS = 8


def synth_waveform(seconds: float = 29.0, seed: int = 0) -> torch.Tensor:
    """A chord progression with noise - enough structure for chroma to latch onto."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    signal = np.zeros_like(t)
    # C major -> A minor -> F major -> G major, repeating, so repeated segments
    # genuinely resemble one another and similarity edges have something to find.
    progression = [[261.63, 329.63, 392.00], [220.00, 261.63, 329.63],
                   [174.61, 220.00, 261.63], [196.00, 246.94, 293.66]]
    bar = len(t) // 8
    for i in range(8):
        chord = progression[i % len(progression)]
        segment = slice(i * bar, (i + 1) * bar)
        for freq in chord:
            signal[segment] += np.sin(2 * np.pi * freq * t[segment]) / len(chord)
    signal += 0.02 * rng.standard_normal(len(t))
    return torch.from_numpy(signal.astype(np.float32))


def build_graphs(n: int = 6) -> list:
    extractor = FeatureExtractor(
        sample_rate=SR, n_fft=AUDIO_CFG["n_fft"], hop_length=AUDIO_CFG["hop_length"],
        n_mels=AUDIO_CFG["n_mels"], n_chroma=AUDIO_CFG["n_chroma"],
        fmin=AUDIO_CFG["fmin"], fmax=AUDIO_CFG["fmax"], device="cpu",
    )
    graph_cfg = GraphConfig(similarity_threshold=0.5, max_similarity_neighbors=4)
    graphs = []
    rng = np.random.default_rng(0)
    for i in range(n):
        graph = build_track_graph(synth_waveform(seed=i), extractor, AUDIO_CFG, graph_cfg)
        graph.y = torch.from_numpy(rng.integers(0, 2, NUM_TAGS).astype(np.float32)).unsqueeze(0)
        graph.input_ids = torch.randint(1, 1000, (1, 32))
        graph.attention_mask = torch.ones(1, 32, dtype=torch.long)
        graph.track_group = torch.tensor([i % 3], dtype=torch.long)
        graphs.append(graph)
    return graphs


def test_feature_extraction() -> None:
    extractor = FeatureExtractor(sample_rate=SR, device="cpu")
    waveform = synth_waveform()
    log_mel, chroma = extractor.log_mel(waveform), extractor.chroma(waveform)

    assert log_mel.shape[0] == 128, f"expected 128 mel bands, got {log_mel.shape[0]}"
    assert chroma.shape[0] == 12, f"expected 12 chroma bins, got {chroma.shape[0]}"
    assert torch.isfinite(log_mel).all() and torch.isfinite(chroma).all()
    # Chroma is L1-normalised per frame.
    assert torch.allclose(chroma.sum(dim=0), torch.ones(chroma.shape[1]), atol=1e-4)

    segments, times = segment_features(
        torch.cat([log_mel, chroma]), SR, 512, 5.0, 0.5, max_segments=16
    )
    pooled = pool_segments(segments)
    assert pooled.shape[0] == segments.shape[0] == len(times)
    assert pooled.shape[1] == 2 * (128 + 12)

    chords = estimate_chords(chroma)
    assert len(chords) > 0, "chord estimation returned an empty sequence"
    print(f"  features OK | mel {tuple(log_mel.shape)} chroma {tuple(chroma.shape)} "
          f"| {pooled.shape[0]} segments | {len(chords)} chord events, e.g. {chords[:4]}")


def test_graph_construction() -> None:
    graphs = build_graphs(3)
    summary = graph_summary(graphs[0])
    assert summary["num_nodes"] > 0 and summary["num_edges"] > 0
    assert summary["num_segments"] > 1
    assert graphs[0].edge_index.max() < graphs[0].num_nodes, "edge index out of range"
    assert graphs[0].x.shape[1] == 2 * (128 + 12)
    print(f"  graph OK | {summary}")


def test_models_forward_and_backward() -> None:
    from torch_geometric.loader import DataLoader as GeoDataLoader

    graphs = build_graphs(6)
    batch = next(iter(GeoDataLoader(graphs, batch_size=3, shuffle=False)))
    node_dim = graphs[0].x.shape[1]

    gnn = GNNTagClassifier(in_dim=node_dim, num_tags=NUM_TAGS, hidden_dim=64, num_layers=2)
    logits = gnn(batch)
    assert logits.shape == (3, NUM_TAGS), logits.shape
    logits.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in gnn.parameters())

    gat = GNNTagClassifier(in_dim=node_dim, num_tags=NUM_TAGS, hidden_dim=64, num_layers=2,
                           conv="gat", heads=4)
    assert gat(batch).shape == (3, NUM_TAGS)

    cnn = MelCNNBaseline(num_tags=NUM_TAGS)
    assert cnn(torch.randn(2, 128, 256)).shape == (2, NUM_TAGS)
    print(f"  models OK | GNN out {tuple(logits.shape)} | node dim {node_dim}")


def test_fusion_modes() -> None:
    from torch_geometric.loader import DataLoader as GeoDataLoader

    graphs = build_graphs(4)
    batch = next(iter(GeoDataLoader(graphs, batch_size=4, shuffle=False)))
    node_dim = graphs[0].x.shape[1]

    gnn_cfg = {"hidden_dim": 64, "num_layers": 2, "conv": "sage", "dropout": 0.1,
               "heads": 4, "readout": "mean_max"}
    # Google's 2-layer/128-hidden BERT: a real checkpoint small enough to keep
    # the smoke test fast, unlike community "tiny" forks whose configs lack the
    # model_type key that transformers 5.x requires.
    text_cfg = {"model_name": "google/bert_uncased_L-2_H-128_A-2", "freeze_bert": False, "unfreeze_last_n": 1}
    criterion = MultiTaskLoss()

    for mode in ("gnn_only", "concat", "cross_attention"):
        fusion_cfg = {"mode": mode, "hidden_dim": 64, "num_heads": 4, "dropout": 0.1}
        model = GNNBertFusion(node_dim, NUM_TAGS, gnn_cfg, fusion_cfg, text_cfg)
        outputs = model(batch)
        assert outputs["tag_logits"].shape == (4, NUM_TAGS), (mode, outputs["tag_logits"].shape)
        loss, parts = criterion(outputs, {"tags": batch.y.view(4, NUM_TAGS)})
        loss.backward()
        assert np.isfinite(parts["total_loss"])
        extra = ""
        if mode == "cross_attention":
            assert "text_attention" in outputs
            extra = f", attention {tuple(outputs['text_attention'].shape)}"
        print(f"  fusion '{mode}' OK | loss {parts['total_loss']:.4f}{extra}")


def test_contrastive_and_metrics() -> None:
    torch.manual_seed(0)
    graph_emb = torch.nn.functional.normalize(torch.randn(12, 32), dim=1)
    text_emb = torch.nn.functional.normalize(torch.randn(12, 32), dim=1)
    groups = np.array([i % 4 for i in range(12)])

    mask = build_duplicate_mask(groups)
    assert mask.shape == (12, 12) and not mask.diagonal().any()
    loss, parts = info_nce_loss(graph_emb, text_emb, 0.07, mask)
    assert torch.isfinite(loss) and np.isfinite(parts["total_loss"])

    # Perfect alignment must give R@1 = 1.
    perfect = retrieval_metrics(graph_emb, graph_emb, ks=(1, 5), group_ids=None)
    assert perfect["caption_to_audio_R@1"] == 1.0, perfect

    scored = retrieval_metrics(graph_emb, text_emb, ks=(1, 5, 10), group_ids=groups)
    assert 0.0 <= scored["caption_to_audio_R@5"] <= 1.0

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, (200, NUM_TAGS)).astype(np.float32)
    y_score = np.clip(y_true * 0.6 + rng.random((200, NUM_TAGS)) * 0.4, 0, 1)
    thresholds = search_thresholds(y_true, y_score)
    metrics = tagging_metrics(y_true, y_score, thresholds, [f"tag{i}" for i in range(NUM_TAGS)])
    assert 0.0 <= metrics["macro_f1"] <= 1.0 and "per_tag" in metrics

    coherence = graph_coherence_score(torch.randn(10, 16), torch.randint(0, 10, (2, 30)))
    assert np.isfinite(coherence)
    print(f"  contrastive+metrics OK | InfoNCE {parts['total_loss']:.4f} | "
          f"macro-F1 {metrics['macro_f1']:.3f} | coherence {coherence:.3f}")


def test_context_strings() -> None:
    text = build_context_string("Kyrie", "American Bach Soloists", "Haydn Masses", ["baroque", "choir"])
    assert "Kyrie" in text and "baroque" in text
    assert build_context_string(None, None, None, None) != ""
    # Auxiliary tags must be omitted when disabled, so Task 1 stays non-circular.
    without = build_context_string("A", "B", "C", ["x"], use_auxiliary_tags=False)
    assert "x" not in without
    print(f"  context strings OK | example: {text[:80]}...")


def main() -> None:
    tests = [
        ("feature extraction", test_feature_extraction),
        ("graph construction", test_graph_construction),
        ("model forward/backward", test_models_forward_and_backward),
        ("fusion modes", test_fusion_modes),
        ("contrastive + metrics", test_contrastive_and_metrics),
        ("context strings", test_context_strings),
    ]
    failed = 0
    for name, fn in tests:
        print(f"[ RUN ] {name}")
        try:
            fn()
            print(f"[ OK  ] {name}\n")
        except Exception as exc:
            failed += 1
            print(f"[FAIL ] {name}: {type(exc).__name__}: {exc}\n")
            import traceback

            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
