"""Shared prediction session for evaluation and interactive example histories.

A session owns causal visit memory. Reset between histories; serialize access
when using the same session from concurrent service requests.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional
import numpy as np
import torch

from reta.knowledge.pool import KnowledgePool
from .runtime import (
    EntityTokenMapper,
    PoolAlignedTokenIndex,
    build_checkpoint_contract,
    encoder_code_embedding_lookup,
    fold_policy_state_for_retrieval,
    load_checkpoint,
    load_state_dict_strict,
    load_processed_data,
    require_checkpoint_contract,
    resolve_device,
    set_seed,
    template_to_subgraph,
    template_vector_tensor,
    visit_retrieval_tokens,
)
from .model import (
    DecoupledMultiGATEncoder,
    text_embedding_kwargs,
    NextVisitPredictor,
    build_pyg_data,
    graft_hard_import,
)
from .policy import (
    HARD,
    SOFT,
    PolicyNet,
    StateEncoder,
    TemplateUtilityTracker,
    build_policy_state,
    decode_policy_action,
    valid_action_mask,
)


def _config_value(config, section: str, key: str, default):
    """Read one value from a format-v2 checkpoint configuration."""
    section_value = config.get(section, {}) if isinstance(config, dict) else {}
    return section_value.get(key, default) if isinstance(section_value, dict) else default


@dataclass
class PredictionSession:
    encoder: DecoupledMultiGATEncoder
    predictor: NextVisitPredictor
    baseline_encoder: Optional[DecoupledMultiGATEncoder]
    baseline_predictor: Optional[NextVisitPredictor]
    policy_net: Optional[PolicyNet]
    state_enc: Optional[StateEncoder]
    pool: KnowledgePool
    mapper: EntityTokenMapper
    aligned_index: PoolAlignedTokenIndex
    utility_tracker: TemplateUtilityTracker
    dim: int
    policy_K: int
    effective_soft_xi: float
    retrieval_alpha: float
    effective_seed: int
    device_t: torch.device
    icd_vocab_size: int
    ccs_label_vocab_size: int
    ck: Dict
    cfg: Dict
    contract: Dict
    training_split: str
    effective_split_json: Optional[str]
    past_visit_embs: List[torch.Tensor] = field(default_factory=list)
    baseline_past_visit_embs: List[torch.Tensor] = field(default_factory=list)

    def reset(self):
        self.past_visit_embs.clear()
        self.baseline_past_visit_embs.clear()
        if self.state_enc is not None:
            self.state_enc.reset()

    @torch.no_grad()
    def predict_visit(self, visit, deterministic=True, trace=False):
        """Predict from observed codes only; labels are never read."""
        from torch_geometric.data import Batch
        encoder = self.encoder
        predictor = self.predictor
        baseline_encoder = self.baseline_encoder
        baseline_predictor = self.baseline_predictor
        policy_net = self.policy_net
        state_enc = self.state_enc
        pool = self.pool
        mapper = self.mapper
        utility_tracker = self.utility_tracker
        dim = self.dim
        policy_K = self.policy_K
        effective_soft_xi = self.effective_soft_xi
        retrieval_alpha = self.retrieval_alpha
        device_t = self.device_t
        past_visit_embs = self.past_visit_embs
        baseline_past_visit_embs = self.baseline_past_visit_embs
        code_embed_lookup = encoder_code_embedding_lookup(encoder)
        # raw graph
        node_ids = visit["node_ids"]
        edge_index = torch.tensor(visit["edge_index"], dtype=torch.long)
        orig_mask = torch.ones(len(node_ids), dtype=torch.bool)
        code_set = set(int(x) for x in visit["icd_tokens"])
        code_mask = torch.tensor([int(nid) in code_set for nid in node_ids], dtype=torch.bool)
        g_raw = build_pyg_data(node_ids, edge_index, orig_mask=orig_mask, code_mask=code_mask).to(device_t)

        soft_vec = None
        g_aug = g_raw
        added_nodes = 0
        action_name = "skip"
        template_id = None
        baseline_h_current = None
        candidates = []
        candidate_scores = []
        raw_logits = None
        mem = (
            torch.stack(past_visit_embs, dim=0)
            if past_visit_embs
            else torch.zeros((0, dim), device=device_t)
        )

        if policy_net is not None and state_enc is not None:
            # Encode the unaugmented visit first so retrieval uses the
            # same learned history/current state as the policy.
            bg_raw = Batch.from_data_list([g_raw]).to(device_t)
            h_G_raw, _ = encoder(bg_raw, past_visit_memory=[mem])
            if trace:
                raw_logits = predictor(h_G_raw)
            if baseline_encoder is not None and baseline_predictor is not None:
                base_mem = (
                    torch.stack(baseline_past_visit_embs, dim=0)
                    if len(baseline_past_visit_embs) > 0
                    else torch.zeros((0, dim), device=device_t)
                )
                h_G_base, _ = baseline_encoder(bg_raw, past_visit_memory=[base_mem])
                base_logits = baseline_predictor(h_G_base)
                baseline_h_current = h_G_base[0].detach()
            else:
                h_G_base = h_G_raw
                base_logits = predictor(h_G_raw)
            uncertainty = 1.0 - torch.sigmoid(base_logits).max()
            s_hist_cur = state_enc.step(h_G_raw[0])
            query_tokens = visit_retrieval_tokens(visit)
            retrieval_state = fold_policy_state_for_retrieval(
                s_hist_cur,
                dim,
            )
            if query_tokens:
                ret = pool.retrieve_topk(
                    query_tokens,
                    code_embed_lookup,
                    K=policy_K,
                    state_vector=retrieval_state,
                    alpha=retrieval_alpha,
                )
                candidates = ret.template_ids
                candidate_scores = ret.scores
            util_vec = utility_tracker.vector(
                candidates,
                K=policy_K,
                device=device_t,
            )
            policy_state = build_policy_state(
                s_hist_cur.to(device_t),
                uncertainty.detach(),
                util_vec,
            )
            action_mask = valid_action_mask(
                len(candidates),
                policy_K,
                device=device_t,
            )
            a, _, _ = policy_net.act(
                policy_state,
                deterministic=deterministic,
                action_mask=action_mask,
            )
            act = decode_policy_action(int(a.item()), candidates, policy_K)

            if act is not None and act.is_skip:
                action_name = "skip"
            elif act is not None:
                tmpl = pool.get_template(act.template_id)
                template_id = int(act.template_id)

                if act.mode == SOFT:
                    action_name = "soft"
                    soft_vec = template_vector_tensor(tmpl, dim)
                elif act.mode == HARD:
                    action_name = "hard"
                    sub = template_to_subgraph(tmpl.medoid.to_dict(), mapper)
                    if sub is not None:
                        before = int(g_raw.node_ids.numel())
                        g_aug = graft_hard_import(g_raw, sub)
                        after = int(g_aug.node_ids.numel())
                        added_nodes = max(0, after - before)
                        if g_aug is g_raw:
                            action_name = "hard_noop"
                    else:
                        action_name = "hard_noop"

        # encode augmented and predict
        bg = Batch.from_data_list([g_aug]).to(device_t)
        soft_offset = soft_vec.view(1, -1).to(device_t) if soft_vec is not None else None
        h_G, _ = encoder(
            bg,
            past_visit_memory=[mem],
            soft_offset_per_graph=soft_offset,
            xi=effective_soft_xi,
        )
        logits = predictor(h_G)

        past_visit_embs.append(h_G[0].detach())
        if baseline_h_current is not None:
            baseline_past_visit_embs.append(baseline_h_current)
        details = {"action": action_name, "template_id": template_id,
                   "added_nodes": int(added_nodes), "used_soft": soft_vec is not None}
        if trace:
            details.update({
                "candidates": [{"template_id": int(tid), "score": float(score)}
                               for tid, score in zip(candidates, candidate_scores)],
                "raw_probabilities": torch.sigmoid(raw_logits if raw_logits is not None else logits)[0].cpu().tolist(),
                "node_ids": g_aug.node_ids.cpu().tolist(),
                "edge_index": g_aug.edge_index.cpu().tolist(),
                "original_node_ids": g_raw.node_ids.cpu().tolist(),
            })
        return logits.detach().cpu(), details


def load_prediction_session(checkpoint_path, processed_path, templates_jsonl,
                            entity_to_token_json=None, split_json=None, device="auto",
                            retrieval_K=None, soft_xi=None, seed=None, data=None):
    """Load once and validate the original training artifact fingerprints."""
    if data is None:
        data = load_processed_data(processed_path)
    meta = data["meta"]
    icd_vocab_size = int(meta["icd_vocab_size"])
    ccs_vocab_size = int(meta["ccs_vocab_size"])
    ccs_label_vocab_size = int(meta.get("ccs_label_vocab_size", ccs_vocab_size))
    base_vocab_size = icd_vocab_size + ccs_vocab_size
    pool = KnowledgePool.load_jsonl(templates_jsonl)
    ck = load_checkpoint(checkpoint_path)
    cfg = ck.get("cfg", None)
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint cfg must be a format-v2 configuration mapping.")
    encoder_state = ck.get("encoder")
    predictor_state = ck.get("predictor")
    if not isinstance(encoder_state, Mapping) or not isinstance(predictor_state, Mapping):
        raise ValueError("checkpoint must contain encoder and predictor state dictionaries.")
    embedding_weight = encoder_state.get("embed.emb.weight")
    if embedding_weight is None or embedding_weight.ndim != 2:
        raise ValueError("checkpoint encoder state is missing embed.emb.weight.")
    projection_weight = encoder_state.get("embed.projection.weight")
    inferred_dim = int(projection_weight.shape[0] if projection_weight is not None else embedding_weight.shape[1])
    dim = int(_config_value(cfg, "model", "dim", inferred_dim))
    gnn_layers = int(_config_value(cfg, "model", "gnn_layers", 2))
    attn_heads = int(_config_value(cfg, "model", "attn_heads", 4))
    dropout = float(_config_value(cfg, "model", "dropout", 0.3))
    effective_retrieval_K = int(
        retrieval_K
        if retrieval_K is not None
        else _config_value(cfg, "knowledge", "retrieval_K", 20)
    )
    effective_soft_xi = float(
        soft_xi
        if soft_xi is not None
        else _config_value(cfg, "train", "soft_xi", 0.5)
    )
    effective_seed = int(
        seed if seed is not None else _config_value(cfg, "train", "seed", 42)
    )
    effective_entity_mapping = entity_to_token_json
    if effective_entity_mapping is None:
        effective_entity_mapping = _config_value(
            cfg,
            "knowledge",
            "entity_to_token_json",
            None,
        )
    effective_split_json = split_json
    if effective_split_json is None:
        effective_split_json = _config_value(cfg, "data", "split_json", None)
    training_split = (
        str(_config_value(cfg, "data", "train_split", "train"))
        if effective_split_json
        else "all"
    )
    if effective_retrieval_K <= 0:
        raise ValueError("retrieval_K must be positive.")
    if effective_soft_xi < 0.0:
        raise ValueError("soft_xi must be non-negative.")
    device_t = resolve_device(device)
    set_seed(effective_seed)
    mapper = EntityTokenMapper.from_sources(
        data.get("vocab", {}).get("name_to_token"),
        effective_entity_mapping,
        base_vocab_size=base_vocab_size,
    )
    aligned_index = PoolAlignedTokenIndex(pool, mapper)
    contract = build_checkpoint_contract(
        mapper=mapper,
        processed_path=processed_path,
        templates_path=templates_jsonl,
        split_json=effective_split_json,
        training_split=training_split,
        icd_vocab_size=icd_vocab_size,
        ccs_vocab_size=ccs_vocab_size,
        ccs_label_vocab_size=ccs_label_vocab_size,
        model_dim=dim,
        gnn_layers=gnn_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        pool_dim=aligned_index.dim,
        retrieval_K=int(_config_value(cfg, "knowledge", "retrieval_K", 20)),
        retrieval_alpha=float(
            _config_value(cfg, "knowledge", "retrieval_alpha", 0.3)
        ),
        soft_xi=float(_config_value(cfg, "train", "soft_xi", 0.5)),
        concept_features_sha256=ck.get("contract", {}).get("concept_features_sha256"),
    )
    require_checkpoint_contract(ck, contract)

    encoder = DecoupledMultiGATEncoder(
        vocab_size=mapper.vocab_size,
        dim=dim,
        gnn_layers=gnn_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        **text_embedding_kwargs(encoder_state),
    ).to(device_t)
    predictor = NextVisitPredictor(
        in_dim=dim,
        num_labels=ccs_label_vocab_size,
        dropout=dropout,
    ).to(device_t)

    load_state_dict_strict(encoder, encoder_state, "encoder")
    load_state_dict_strict(predictor, predictor_state, "predictor")

    baseline_encoder = None
    baseline_predictor = None
    if ("baseline_encoder" in ck) != ("baseline_predictor" in ck):
        raise ValueError("checkpoint must contain both baseline components or neither.")
    if "baseline_encoder" in ck and "baseline_predictor" in ck:
        baseline_encoder = DecoupledMultiGATEncoder(
            vocab_size=mapper.vocab_size,
            dim=dim,
            gnn_layers=gnn_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            **text_embedding_kwargs(ck["baseline_encoder"]),
        ).to(device_t)
        baseline_predictor = NextVisitPredictor(
            in_dim=dim,
            num_labels=ccs_label_vocab_size,
            dropout=dropout,
        ).to(device_t)
        load_state_dict_strict(
            baseline_encoder,
            ck["baseline_encoder"],
            "baseline encoder",
        )
        load_state_dict_strict(
            baseline_predictor,
            ck["baseline_predictor"],
            "baseline predictor",
        )

    # policy optional (if rl checkpoint)
    policy_net = None
    state_enc = None
    policy_K = effective_retrieval_K
    utility_tracker = TemplateUtilityTracker()
    utility_values = ck.get("template_utilities", {})
    if not isinstance(utility_values, Mapping):
        raise ValueError("checkpoint template_utilities must be a mapping.")
    parsed_utilities = {}
    known_template_ids = set(pool.id_to_index)
    for raw_template_id, raw_value in utility_values.items():
        template_id = int(raw_template_id)
        value = float(raw_value)
        if template_id not in known_template_ids:
            raise ValueError(
                f"checkpoint utility references unknown template {template_id}."
            )
        if template_id in parsed_utilities:
            raise ValueError(
                f"checkpoint contains duplicate utility key for template {template_id}."
            )
        if not np.isfinite(value):
            raise ValueError(
                f"checkpoint utility for template {template_id} is not finite."
            )
        parsed_utilities[template_id] = value
    utility_tracker.values = parsed_utilities
    retrieval_alpha = float(_config_value(cfg, "knowledge", "retrieval_alpha", 0.3))
    if "policy" in ck:
        if baseline_encoder is None or baseline_predictor is None:
            raise ValueError("policy checkpoint is missing its frozen baseline models.")
        policy_action_dim = int(ck.get("action_dim", 0))
        if policy_action_dim <= 0 or policy_action_dim % 2 != 1:
            raise ValueError("checkpoint policy action_dim must be a positive odd integer.")
        policy_K = (policy_action_dim - 1) // 2
        checkpoint_K = int(_config_value(cfg, "knowledge", "retrieval_K", 0))
        if policy_K != checkpoint_K:
            raise ValueError(
                f"checkpoint action_dim implies K={policy_K}, but config stores K={checkpoint_K}."
            )
        if retrieval_K is not None and int(retrieval_K) != policy_K:
            raise ValueError(
                f"retrieval_K={retrieval_K} conflicts with checkpoint policy K={policy_K}."
            )
        policy_state_dim = int(ck.get("policy_state_dim", 0))
        expected_state_dim = 2 * dim + 1 + policy_K
        if policy_state_dim != expected_state_dim:
            raise ValueError(
                f"checkpoint policy_state_dim={policy_state_dim}, expected {expected_state_dim}."
            )
        policy_net = PolicyNet(state_dim=policy_state_dim, action_dim=policy_action_dim, hidden_dim=256).to(device_t)
        load_state_dict_strict(policy_net, ck["policy"], "policy")
        policy_net.eval()
        state_enc = StateEncoder(dim=dim).to(device_t)
        if "state_encoder" not in ck:
            raise ValueError("policy checkpoint is missing state_encoder.")
        load_state_dict_strict(state_enc, ck["state_encoder"], "state encoder")
        state_enc.eval()

    encoder.eval()
    predictor.eval()
    if baseline_encoder is not None and baseline_predictor is not None:
        baseline_encoder.eval()
        baseline_predictor.eval()

    return PredictionSession(
        encoder=encoder,
        predictor=predictor,
        baseline_encoder=baseline_encoder,
        baseline_predictor=baseline_predictor,
        policy_net=policy_net,
        state_enc=state_enc,
        pool=pool,
        mapper=mapper,
        aligned_index=aligned_index,
        utility_tracker=utility_tracker,
        dim=dim,
        policy_K=policy_K,
        effective_soft_xi=effective_soft_xi,
        retrieval_alpha=retrieval_alpha,
        effective_seed=effective_seed,
        device_t=device_t,
        icd_vocab_size=icd_vocab_size,
        ccs_label_vocab_size=ccs_label_vocab_size,
        ck=ck,
        cfg=cfg,
        contract=contract,
        training_split=training_split,
        effective_split_json=effective_split_json,
    )
