"""Construction of per-track music structure graphs.

Each track becomes one heterogeneous-but-flat PyG ``Data`` object containing:

  * **segment nodes** - fixed-length windows of the track, featurised by
    mean/std-pooled log-mel and chroma.
  * **chord nodes** - the unique template-matched triads occurring in the track.

and four kinds of edge, distinguished by ``edge_type``:

  0. segment -> segment, temporal adjacency (both directions)
  1. segment -> segment, cosine-similarity above threshold (repetition structure)
  2. chord   -> chord,   observed transitions, weighted by occurrence count
  3. segment <-> chord,  membership (which chords sound during which segment)

Keeping the graph flat (one node feature matrix, one edge index) means stock
``SAGEConv`` / ``GATConv`` layers work unchanged, while ``node_type`` and
``edge_type`` remain available for analysis and ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from .audio_features import (
    CHORD_NAMES,
    CHORD_TEMPLATES,
    FeatureExtractor,
    estimate_chords,
    pool_segments,
    segment_features,
)

EDGE_TEMPORAL = 0
EDGE_SIMILARITY = 1
EDGE_CHORD_TRANSITION = 2
EDGE_MEMBERSHIP = 3

NODE_SEGMENT = 0
NODE_CHORD = 1


@dataclass
class GraphConfig:
    similarity_threshold: float = 0.85
    max_similarity_neighbors: int = 4
    add_self_loops: bool = True
    include_chord_graph: bool = True
    chord_min_frames: int = 4


def build_track_graph(
    waveform: torch.Tensor,
    extractor: FeatureExtractor,
    audio_cfg,
    graph_cfg: GraphConfig,
) -> Data:
    """Turn one waveform into a music structure graph."""
    log_mel = extractor.log_mel(waveform)   # (n_mels, frames)
    chroma = extractor.chroma(waveform)     # (12, frames)

    # Align frame counts; the two transforms can differ by a frame at the edges.
    n_frames = min(log_mel.shape[1], chroma.shape[1])
    log_mel, chroma = log_mel[:, :n_frames], chroma[:, :n_frames]

    stacked = torch.cat([log_mel, chroma], dim=0).cpu()
    segments, times = segment_features(
        stacked,
        sample_rate=audio_cfg["sample_rate"],
        hop_length=audio_cfg["hop_length"],
        segment_seconds=audio_cfg["segment_seconds"],
        overlap=audio_cfg["segment_overlap"],
        max_segments=audio_cfg["max_segments"],
    )
    seg_feats = pool_segments(segments)  # (n_seg, 2 * (n_mels + 12))
    n_seg = seg_feats.shape[0]

    node_features = [seg_feats]
    node_types = [torch.full((n_seg,), NODE_SEGMENT, dtype=torch.long)]
    edges: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    edge_weights: list[torch.Tensor] = []

    # --- 0. temporal adjacency ------------------------------------------- #
    if n_seg > 1:
        src = torch.arange(n_seg - 1)
        dst = src + 1
        temporal = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        edges.append(temporal)
        edge_types.append(torch.full((temporal.shape[1],), EDGE_TEMPORAL, dtype=torch.long))
        edge_weights.append(torch.ones(temporal.shape[1]))

    # --- 1. similarity edges --------------------------------------------- #
    sim_edges, sim_weights = _similarity_edges(
        seg_feats, graph_cfg.similarity_threshold, graph_cfg.max_similarity_neighbors
    )
    if sim_edges.numel():
        edges.append(sim_edges)
        edge_types.append(torch.full((sim_edges.shape[1],), EDGE_SIMILARITY, dtype=torch.long))
        edge_weights.append(sim_weights)

    # --- 2 & 3. chord graph ----------------------------------------------- #
    chord_sequence: list[str] = []
    if graph_cfg.include_chord_graph:
        chord_sequence = estimate_chords(chroma, min_frames=graph_cfg.chord_min_frames)
        unique_chords = sorted(set(chord_sequence))
        if unique_chords:
            chord_index = {name: n_seg + i for i, name in enumerate(unique_chords)}

            # Chord node features: the triad template broadcast into the same
            # layout as segment features (mel slots carry the track's mean mel
            # profile so the two node types live in a comparable space).
            mel_dim = log_mel.shape[0]
            mel_mean = log_mel.mean(dim=1).cpu()
            mel_std = log_mel.std(dim=1, unbiased=False).cpu()
            chord_feats = []
            for name in unique_chords:
                template = CHORD_TEMPLATES[CHORD_NAMES.index(name)]
                chord_feats.append(
                    torch.cat([mel_mean, mel_std, template, torch.zeros(template.shape[0])])
                )
            chord_feats_t = torch.stack(chord_feats)
            assert chord_feats_t.shape[1] == seg_feats.shape[1], (
                f"chord feature dim {chord_feats_t.shape[1]} != segment dim {seg_feats.shape[1]}"
            )
            node_features.append(chord_feats_t)
            node_types.append(torch.full((len(unique_chords),), NODE_CHORD, dtype=torch.long))

            # Transition edges weighted by observed count.
            transitions: dict[tuple[int, int], float] = {}
            for a, b in zip(chord_sequence[:-1], chord_sequence[1:]):
                key = (chord_index[a], chord_index[b])
                transitions[key] = transitions.get(key, 0.0) + 1.0
            if transitions:
                pairs = torch.tensor(list(transitions.keys()), dtype=torch.long).T
                counts = torch.tensor(list(transitions.values()), dtype=torch.float)
                edges.append(pairs)
                edge_types.append(torch.full((pairs.shape[1],), EDGE_CHORD_TRANSITION, dtype=torch.long))
                edge_weights.append(counts / counts.max())

            # Membership edges: attach every segment to the chords sounding in it.
            member = _membership_edges(
                chroma, times, chord_index, audio_cfg, n_seg, graph_cfg.chord_min_frames
            )
            if member.numel():
                edges.append(member)
                edge_types.append(torch.full((member.shape[1],), EDGE_MEMBERSHIP, dtype=torch.long))
                edge_weights.append(torch.ones(member.shape[1]))

    x = torch.cat(node_features, dim=0)
    n_nodes = x.shape[0]

    if edges:
        edge_index = torch.cat(edges, dim=1)
        edge_type = torch.cat(edge_types)
        edge_weight = torch.cat(edge_weights)
    else:  # degenerate single-segment track
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)

    if graph_cfg.add_self_loops:
        loops = torch.arange(n_nodes).repeat(2, 1)
        edge_index = torch.cat([edge_index, loops], dim=1)
        edge_type = torch.cat([edge_type, torch.full((n_nodes,), EDGE_TEMPORAL, dtype=torch.long)])
        edge_weight = torch.cat([edge_weight, torch.ones(n_nodes)])

    return Data(
        x=x.float(),
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        node_type=torch.cat(node_types),
        num_nodes=n_nodes,
        n_segments=n_seg,
        chord_sequence=",".join(chord_sequence),
        segment_times=torch.from_numpy(times).float(),
    )


def _similarity_edges(
    seg_feats: torch.Tensor, threshold: float, max_neighbors: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine-similarity edges between non-adjacent segments (repetition structure)."""
    n = seg_feats.shape[0]
    if n < 3:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,))

    normed = torch.nn.functional.normalize(seg_feats, dim=1)
    sim = normed @ normed.T

    # Exclude self and immediate temporal neighbours, which the temporal edges
    # already cover and which are trivially similar.
    mask = torch.ones_like(sim, dtype=torch.bool)
    for offset in (-1, 0, 1):
        mask &= ~torch.eye(n, dtype=torch.bool).roll(offset, dims=1)
    sim = sim.masked_fill(~mask, -1.0)

    src_list, dst_list, weight_list = [], [], []
    k = min(max_neighbors, n - 1)
    top_values, top_indices = sim.topk(k, dim=1)
    for i in range(n):
        for value, j in zip(top_values[i], top_indices[i]):
            if value.item() >= threshold:
                src_list.append(i)
                dst_list.append(int(j))
                weight_list.append(float(value))

    if not src_list:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,))

    # Symmetrise so message passing flows both ways.
    src = torch.tensor(src_list + dst_list, dtype=torch.long)
    dst = torch.tensor(dst_list + src_list, dtype=torch.long)
    weights = torch.tensor(weight_list + weight_list, dtype=torch.float)
    return torch.stack([src, dst]), weights


def _membership_edges(
    chroma: torch.Tensor,
    times: np.ndarray,
    chord_index: dict[str, int],
    audio_cfg,
    n_seg: int,
    min_frames: int,
) -> torch.Tensor:
    """Link each segment to the chord nodes whose triad dominates its frames."""
    frames_per_segment = max(
        1, int(round(audio_cfg["segment_seconds"] * audio_cfg["sample_rate"] / audio_cfg["hop_length"]))
    )
    scores = (CHORD_TEMPLATES @ chroma.cpu()).argmax(dim=0).numpy()

    src, dst = [], []
    for seg_i, start_time in enumerate(times[:n_seg]):
        start = int(round(start_time * audio_cfg["sample_rate"] / audio_cfg["hop_length"]))
        window = scores[start : start + frames_per_segment]
        if window.size == 0:
            continue
        # Keep chords occupying at least min_frames of the window.
        values, counts = np.unique(window, return_counts=True)
        for value, count in zip(values, counts):
            if count < min_frames:
                continue
            node = chord_index.get(CHORD_NAMES[int(value)])
            if node is not None:
                src.append(seg_i)
                dst.append(node)

    if not src:
        return torch.empty((2, 0), dtype=torch.long)

    src_t = torch.tensor(src + dst, dtype=torch.long)
    dst_t = torch.tensor(dst + src, dtype=torch.long)
    return torch.stack([src_t, dst_t])


def graph_summary(data: Data) -> dict:
    """Small dict used for the EDA notebook and the exported example graphs."""
    per_type = {
        name: int((data.edge_type == value).sum())
        for name, value in (
            ("temporal", EDGE_TEMPORAL),
            ("similarity", EDGE_SIMILARITY),
            ("chord_transition", EDGE_CHORD_TRANSITION),
            ("membership", EDGE_MEMBERSHIP),
        )
    }
    return {
        "num_nodes": int(data.num_nodes),
        "num_segments": int(data.n_segments),
        "num_chords": int((data.node_type == NODE_CHORD).sum()),
        "num_edges": int(data.edge_index.shape[1]),
        "edges_by_type": per_type,
        "feature_dim": int(data.x.shape[1]),
    }
