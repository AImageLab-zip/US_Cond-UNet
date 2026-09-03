"""Helpers shared by analyze_unet_attn.ipynb and analyze_unet_film.ipynb."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets.segm_net import FiLM2d, UNet2DFiLM  # noqa: E402
from nets.unet_attn import UNet2DAttn  # noqa: E402


# ----------------------------------------------------------------------------- #
# Modulation collection
# ----------------------------------------------------------------------------- #

@torch.no_grad()
def compute_per_sample_dice(logits: torch.Tensor, masks: torch.Tensor,
                            threshold: float = 0.5) -> torch.Tensor:
    """Per-sample binary Dice. logits, masks: (B, H, W)."""
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    if masks.dim() == 4:
        masks = masks.squeeze(1)
    pred = (logits > threshold).float()
    gt = (masks > 0.5).float()
    dims = (1, 2)
    inter = (pred * gt).sum(dim=dims) * 2.0
    denom = pred.sum(dim=dims) + gt.sum(dim=dims)
    return (inter + 1e-6) / (denom + 1e-6)


def _is_attn(model) -> bool:
    return isinstance(model, UNet2DAttn)


def _organ_embed(model) -> nn.Embedding:
    if _is_attn(model):
        return model.shared_attn.organ_embed
    # FiLM model has many FiLM2d, each with its own organ embedding — we don't
    # have a single global organ embedding. Surface this as None.
    return None


@torch.no_grad()
def collect_modulations(model, loader, device,
                        max_batches: int | None = None,
                        run_forward: bool = True) -> dict:
    """Collect compact γ, β per layer + organ_id + (optionally) dice per sample.

    Set run_forward=False to skip the UNet forward pass — sections 4.x, 6.x,
    7.x, 8, 9, 10 only need the modulations; call compute_dice_scores() before
    section 5.2 when you need dice.

    Returns dict with keys:
        "layers": dict[layer_id -> {"gamma": (N, C), "beta": (N, C)}]
        "organ_id": (N,) long
        "dice": (N,) float  — all NaN when run_forward=False
        "layer_configs": [(layer_id, n_channels), ...]
        "max_channels": int
    """
    model.eval()
    layer_configs = model._build_layer_configs()
    n_layers = len(layer_configs)
    max_channels = model.size * (2 ** (model.depth + 1))

    layer_gammas: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    layer_betas: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    organ_ids: list[torch.Tensor] = []
    dices: list[torch.Tensor] = []

    n_done = 0
    for batch in loader:
        if max_batches is not None and n_done >= max_batches:
            break
        n_done += 1
        pixel_values = batch["pixel_values"].to(device)
        organ_id = batch["organ_id"].to(device).long()
        masks = batch["masks"].to(device)

        # Capture compact modulations — also keep the channel-adapted versions so
        # we can inject them into the forward pass without recomputing attention.
        modulations, _, compact = model.shared_attn.compute_all_modulations(
            pixel_values, organ_id, layer_configs, return_compact=True
        )
        for i, (layer_id, _) in enumerate(layer_configs):
            g, b, _ = compact[layer_id]
            layer_gammas[i].append(g.detach().cpu())
            layer_betas[i].append(b.detach().cpu())

        organ_ids.append(organ_id.detach().cpu())

        if run_forward:
            # Reuse already-computed modulations — avoids a second attention pass
            # inside model.forward() for the Attn model.
            mod_list = [modulations[lid] for lid, _ in layer_configs]
            with inject_modulations(model, mod_list):
                out = model(pixel_values=pixel_values, organ_id=organ_id, masks=masks)
            dice = compute_per_sample_dice(out["logits"], masks)
            dices.append(dice.detach().cpu())
        else:
            dices.append(torch.full((pixel_values.shape[0],), float("nan")))

    layers = {}
    for i, (layer_id, n_ch) in enumerate(layer_configs):
        layers[layer_id] = {
            "gamma": torch.cat(layer_gammas[i], dim=0).numpy(),  # (N, max_channels)
            "beta": torch.cat(layer_betas[i], dim=0).numpy(),
            "n_channels": n_ch,
        }
    return {
        "layers": layers,
        "organ_id": torch.cat(organ_ids).numpy(),
        "dice": torch.cat(dices).numpy(),
        "layer_configs": layer_configs,
        "max_channels": max_channels,
    }


@torch.no_grad()
def compute_dice_scores(model, loader, device, max_batches: int | None = None) -> np.ndarray:
    """Run model.forward() and return per-sample Dice — call this before section 5.2
    when collect_modulations was called with run_forward=False."""
    model.eval()
    dices: list[torch.Tensor] = []
    n_done = 0
    for batch in loader:
        if max_batches is not None and n_done >= max_batches:
            break
        n_done += 1
        pixel_values = batch["pixel_values"].to(device)
        organ_id = batch["organ_id"].to(device).long()
        masks = batch["masks"].to(device)
        out = model(pixel_values=pixel_values, organ_id=organ_id, masks=masks)
        dices.append(compute_per_sample_dice(out["logits"], masks).detach().cpu())
    return torch.cat(dices).numpy() if dices else np.array([], dtype=np.float32)


# ----------------------------------------------------------------------------- #
# Tier 1 — descriptive stats
# ----------------------------------------------------------------------------- #

def per_channel_selectivity(layer_data: dict, organ_ids: np.ndarray) -> np.ndarray:
    """S_c = Var_across_organ(mean_c|organ) / mean_across_organ(Var_within_organ).

    layer_data["gamma"]: (N, C). Returns (C,) selectivity per channel.
    """
    gamma = layer_data["gamma"]
    unique_organs = [o for o in np.unique(organ_ids) if o >= 0]
    means, vars_ = [], []
    for o in unique_organs:
        mask = organ_ids == o
        if mask.sum() < 2:
            continue
        means.append(gamma[mask].mean(axis=0))
        vars_.append(gamma[mask].var(axis=0))
    if len(means) < 2:
        return np.zeros(gamma.shape[1])
    means = np.stack(means)  # (n_organs, C)
    vars_ = np.stack(vars_)  # (n_organs, C)
    var_across = means.var(axis=0)
    mean_within = vars_.mean(axis=0)
    return var_across / (mean_within + 1e-12)


def per_organ_mean_gamma(layer_data: dict, organ_ids: np.ndarray) -> dict:
    """Return {organ_id: mean γ vector (C,)} and {organ_id: mean β vector (C,)}."""
    gamma = layer_data["gamma"]
    beta = layer_data["beta"]
    organs = [o for o in np.unique(organ_ids) if o >= 0]
    g_per = {int(o): gamma[organ_ids == o].mean(axis=0) for o in organs}
    b_per = {int(o): beta[organ_ids == o].mean(axis=0) for o in organs}
    return g_per, b_per


# ----------------------------------------------------------------------------- #
# Tier 3 — interventions (context managers)
# ----------------------------------------------------------------------------- #

@contextmanager
def inject_modulations(model, mod_list: list[tuple[torch.Tensor, torch.Tensor]]):
    """Force model to use the given (gamma, beta) per layer for the duration of
    the with-block. mod_list is ordered the same as model._build_layer_configs().

    Each (gamma, beta) is shape (B, n_channels, 1, 1) — must match the layer's
    n_channels. They will be broadcast against batch dim if B=1.
    """
    if _is_attn(model):
        original = model._prepare_forward

        def patched(*args, **kwargs):
            return {
                "mod_list": [(g, b) for (g, b) in mod_list],
                "mod_idx": 0,
                "projected_shapes": None,
            }

        model._prepare_forward = patched
        try:
            yield
        finally:
            model._prepare_forward = original
        return

    # FiLM model: patch each FiLM2d.forward
    if not isinstance(model, UNet2DFiLM):
        raise TypeError(f"Unsupported model type {type(model)}")
    film_layers = model._get_film_layers_in_order()
    if len(film_layers) != len(mod_list):
        raise RuntimeError(
            f"FiLM layer/mod_list mismatch: {len(film_layers)} vs {len(mod_list)}"
        )
    originals = []
    for fl, (g, b) in zip(film_layers, mod_list):
        originals.append(fl.forward)

        def make_patched(_g=g, _b=b):
            def forward(x, organ_id):
                return _g.to(x.device) * x + _b.to(x.device)
            return forward

        fl.forward = make_patched()
    try:
        yield
    finally:
        for fl, fwd in zip(film_layers, originals):
            fl.forward = fwd


def _broadcast_pair(vec_c: np.ndarray, batch_size: int, device) -> torch.Tensor:
    """(C,) numpy → (B, C, 1, 1) tensor."""
    t = torch.as_tensor(vec_c, dtype=torch.float32, device=device)
    return t.view(1, -1, 1, 1).expand(batch_size, -1, -1, -1).contiguous()


def build_mod_list_from_per_organ(
    organ_ids: torch.Tensor,
    g_per_layer_per_organ: list[dict[int, np.ndarray]],
    b_per_layer_per_organ: list[dict[int, np.ndarray]],
    layer_configs: list[tuple[int, int]],
    max_channels: int,
    device,
    fallback_organ: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build a mod_list where each sample's modulation is the mean for its organ.

    g_per_layer_per_organ[layer_idx][organ_id] → (max_channels,) numpy
    layer_configs gives the target n_channels per layer; we adaptive-pool max_channels → n_channels.
    """
    B = organ_ids.shape[0]
    mod_list = []
    for li, (layer_id, n_ch) in enumerate(layer_configs):
        g_dict = g_per_layer_per_organ[li]
        b_dict = b_per_layer_per_organ[li]
        gammas, betas = [], []
        for o in organ_ids.tolist():
            key = int(o) if int(o) in g_dict else (
                fallback_organ if fallback_organ is not None else next(iter(g_dict))
            )
            gammas.append(g_dict[key])
            betas.append(b_dict[key])
        g_max = torch.as_tensor(np.stack(gammas), dtype=torch.float32, device=device)  # (B, max_channels)
        b_max = torch.as_tensor(np.stack(betas), dtype=torch.float32, device=device)
        g = F.adaptive_avg_pool1d(g_max.unsqueeze(1), n_ch).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        b = F.adaptive_avg_pool1d(b_max.unsqueeze(1), n_ch).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        mod_list.append((g, b))
    return mod_list


def build_mod_list_constant(
    batch_size: int,
    layer_configs: list[tuple[int, int]],
    max_channels: int,
    device,
    gamma_val: float = 1.0,
    beta_val: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Constant γ/β across all layers (default identity)."""
    mod_list = []
    for layer_id, n_ch in layer_configs:
        g = torch.full((batch_size, n_ch, 1, 1), gamma_val, device=device)
        b = torch.full((batch_size, n_ch, 1, 1), beta_val, device=device)
        mod_list.append((g, b))
    return mod_list


@torch.no_grad()
def run_with_modulations(model, batch, mod_list, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject mod_list and forward. Returns (logits, dice)."""
    pixel_values = batch["pixel_values"].to(device)
    organ_id = batch["organ_id"].to(device).long()
    masks = batch["masks"].to(device)
    with inject_modulations(model, mod_list):
        out = model(pixel_values=pixel_values, organ_id=organ_id, masks=masks)
    logits = out["logits"]
    dice = compute_per_sample_dice(logits, masks)
    return logits, dice


# ----------------------------------------------------------------------------- #
# Tier 4 — OOD organ embedding strategies (Attn model only — FiLM has per-block
# embeddings so 'inject a random organ embedding' isn't well-defined the same way)
# ----------------------------------------------------------------------------- #

@torch.no_grad()
def attn_modulations_from_embedding(
    model: UNet2DAttn, pixel_values: torch.Tensor, organ_emb: torch.Tensor,
    layer_configs: list[tuple[int, int]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Bypass organ_id lookup; use the given embedding tensor directly.

    organ_emb: (B, D) — the same emb_dim used by shared_attn.organ_embed.
    Replicates compute_all_modulations but injects organ_emb instead of self.organ_embed(idx).
    Returns mod_list compatible with inject_modulations.
    """
    sa = model.shared_attn
    B = pixel_values.shape[0]
    patches = sa.patch_embed(pixel_values)  # (B, N, D)
    mod_list = []
    for layer_id, n_channels in layer_configs:
        layer_emb = sa.layer_embed(torch.tensor([layer_id], device=pixel_values.device))
        layer_emb = layer_emb.expand(B, -1)
        q = organ_emb + layer_emb  # (B, D)
        queries_organ = q.unsqueeze(1)  # (B, 1, D)
        influence = sa.influence_token.expand(B, -1, -1)  # (B, 1, D)
        queries = torch.cat([influence, queries_organ], dim=1)
        attn_out, _ = sa.attn(query=queries, key=patches, value=patches)
        attn_out = attn_out.mean(dim=1)
        x = sa.shared_norm(attn_out)
        x = sa.layer_linears[layer_id](x)
        x = sa.shared_gelu(x)
        x = sa.shared_dropout(x)
        gb = sa.shared_output(x)
        beta_max, gamma_max = gb.chunk(2, dim=-1)
        beta = F.adaptive_avg_pool1d(beta_max, n_channels).unsqueeze(-1).unsqueeze(-1)
        gamma = F.adaptive_avg_pool1d(gamma_max, n_channels).unsqueeze(-1).unsqueeze(-1)
        mod_list.append((gamma, beta))
    return mod_list


@torch.no_grad()
def attn_modulations_with_organ_arithmetic(
    model: UNet2DAttn, pixel_values: torch.Tensor, organ_emb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward the Attn model using an injected organ embedding; returns (logits, dice_or_none).

    organ_emb: (B, D)
    """
    layer_configs = model._build_layer_configs()
    mod_list = attn_modulations_from_embedding(model, pixel_values, organ_emb, layer_configs)
    return mod_list


@torch.no_grad()
def film_modulations_from_embedding(
    model: UNet2DFiLM, embedding: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """For each FiLM2d block, run its MLP on the injected embedding to produce γ/β.

    embedding: (B, film_embed). Same vector is reinterpreted by each block's MLP.
    Returns mod_list of (γ, β) shaped (B, n_channels, 1, 1) per layer.
    """
    film_layers = model._get_film_layers_in_order()
    mod_list = []
    for fl in film_layers:
        beta_gamma = fl.mlp(embedding)
        beta, gamma = beta_gamma.chunk(2, dim=-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        mod_list.append((gamma, beta))
    return mod_list


# ----------------------------------------------------------------------------- #
# OOD sampling utilities (works for both models — the FiLM model has separate
# embeddings per FiLM2d block; we sample once and apply uniformly per the same
# principle, but interpretability is limited there)
# ----------------------------------------------------------------------------- #

def empirical_gaussian_params(weight: torch.Tensor):
    """Return (mu, L) so samples = mu + L @ z, z ~ N(0, I)."""
    w = weight.detach().cpu().numpy()
    mu = w.mean(axis=0)
    cov = np.cov(w, rowvar=False)
    # add jitter for PD
    cov += 1e-6 * np.eye(cov.shape[0])
    L = np.linalg.cholesky(cov)
    return mu, L


def sample_init_prior(D: int, B: int, std: float = 0.02, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, std, size=(B, D)).astype(np.float32)


def sample_empirical_posterior(mu: np.ndarray, L: np.ndarray, B: int,
                               seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    D = mu.shape[0]
    z = rng.normal(size=(B, D)).astype(np.float32)
    return mu[None, :] + z @ L.T


def nearest_organ_by_cosine(samples: np.ndarray, organ_embed: np.ndarray) -> np.ndarray:
    """For each sample row, find index of nearest organ embedding row by cosine sim."""
    s = samples / (np.linalg.norm(samples, axis=1, keepdims=True) + 1e-12)
    o = organ_embed / (np.linalg.norm(organ_embed, axis=1, keepdims=True) + 1e-12)
    sims = s @ o.T  # (B, n_organs)
    return sims.argmax(axis=1)


# ----------------------------------------------------------------------------- #
# Probes / TCAV
# ----------------------------------------------------------------------------- #

def fit_linear_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    mask = y >= 0
    X, y = X[mask], y[mask]
    if len(np.unique(y)) < 2:
        return {"acc": float("nan"), "n_classes": int(len(np.unique(y)))}
    skf = StratifiedKFold(n_splits=min(n_splits, np.bincount(y).min()), shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, multi_class="auto").fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return {"acc": float(np.mean(accs)), "std": float(np.std(accs)), "n_classes": int(len(np.unique(y)))}


def fit_dice_regressor(X: np.ndarray, dice: np.ndarray) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    if np.allclose(X.std(axis=0), 0):
        return {"r2": float("nan"), "note": "X has zero variance (image-invariant modulation)"}
    sc = StandardScaler().fit(X)
    scores = cross_val_score(Ridge(alpha=1.0), sc.transform(X), dice, cv=5, scoring="r2")
    return {"r2": float(np.mean(scores)), "std": float(np.std(scores))}


def compute_cav(X: np.ndarray, y_binary: np.ndarray) -> np.ndarray:
    """y_binary: 1 = concept (e.g., organ_k), 0 = other. Returns normalized CAV (D,)."""
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LinearSVC(C=1.0, max_iter=5000).fit(sc.transform(X), y_binary)
    cav = clf.coef_[0]
    cav = cav / (np.linalg.norm(cav) + 1e-12)
    return cav


# ----------------------------------------------------------------------------- #
# PCA / GANSpace-style helpers
# ----------------------------------------------------------------------------- #

def pca_on_gammas(
    layers: dict,
    organ_ids: np.ndarray,
    n_components: int = 4,
    normalize: bool = True,
    per_sample: bool = False,
):
    """GANSpace-style PCA over γ vectors concatenated across layers.

    Uses numpy SVD directly (sklearn's PCA on n_features >> n_samples can hang
    on some sklearn versions and computes more than we need).

    Args:
        layers: data['layers'] dict {layer_id: {"gamma": (N, C), ...}}.
        organ_ids: (N,) organ id per sample (used for grouping & color labels).
        n_components: number of PCs to keep.
        normalize: if True, z-score each layer's γ block independently before
            concatenation. Prevents layers with large γ magnitude (or many
            channels) from dominating the PCA, which is essential when the
            Attn model has γ ~ O(10²) at some layers and O(1) at others.
        per_sample: if False (default), PCA over per-organ-mean γ → (n_organs, k).
            If True, PCA over individual sample γ → (N, k). The first is what you
            want for organ-level GANSpace; the second to see within-organ spread
            (only meaningful for the Attn model — FiLM rows collapse per organ).

    Returns:
        proj: (M, k) projections (M = n_organs or N depending on per_sample)
        result: object with .components_ (k, total_dim) and .explained_variance_ratio_
        groups: list of organ_ids — len == M.
    """
    unique_organs = sorted([int(o) for o in np.unique(organ_ids) if o >= 0])
    layer_keys = sorted(layers.keys())

    if per_sample:
        valid = organ_ids >= 0
        blocks = []
        for lid in layer_keys:
            blk = layers[lid]["gamma"][valid].astype(np.float32)
            if normalize:
                mu = blk.mean(axis=0, keepdims=True)
                sd = blk.std(axis=0, keepdims=True) + 1e-6
                blk = (blk - mu) / sd
            blocks.append(blk)
        X = np.concatenate(blocks, axis=1)               # (N_valid, total_dim)
        groups = organ_ids[valid].astype(int).tolist()
    else:
        rows = []
        for o in unique_organs:
            mask = organ_ids == o
            if mask.sum() == 0:
                continue
            chunks = []
            for lid in layer_keys:
                blk = layers[lid]["gamma"][mask].mean(axis=0).astype(np.float32)
                chunks.append(blk)
            rows.append(np.concatenate(chunks))
        X = np.stack(rows)                                # (n_organs, total_dim)
        if normalize:
            # column-wise z-score across organs
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True) + 1e-6
            X = (X - mu) / sd
        groups = unique_organs

    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = max(1, min(n_components, S.size))
    proj = U[:, :k] * S[:k]
    evr_full = (S ** 2) / max(float((S ** 2).sum()), 1e-12)

    class _Result:
        pass

    res = _Result()
    res.components_ = Vt[:k]
    res.explained_variance_ratio_ = evr_full[:k]
    res.singular_values_ = S[:k]
    return proj, res, groups
