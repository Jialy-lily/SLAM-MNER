import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3, hidden_dim: int = 1024, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slot_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor):
        b, n, d = inputs.shape
        inputs = self.norm_inputs(inputs)
        mu = self.slot_mu.expand(b, self.num_slots, -1)
        sigma = torch.exp(self.slot_logsigma).expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        k = self.to_k(inputs)
        v = self.to_v(inputs)
        mask = input_mask.unsqueeze(1)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)
            attn_logits = torch.einsum("bsd,bnd->bsn", q, k) * self.scale
            attn_logits = attn_logits.masked_fill(mask == 0, -1e4)
            attn = F.softmax(attn_logits, dim=-1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bsn,bnd->bsd", attn, v)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            ).reshape(b, self.num_slots, d)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class MultiImageAtomEncoder(nn.Module):
    def __init__(self, vision_name: str, hidden_size: int, pooled_hw: int = 4):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(
            vision_name,
            local_files_only=False,
            use_safetensors=False,
        )
        self.hidden_size = hidden_size
        self.pooled_hw = pooled_hw
        self.proj = nn.Linear(self.vision.config.hidden_size, hidden_size)

    def forward(self, pixel_values: torch.Tensor, image_mask: torch.Tensor):
        b, m = pixel_values.shape[:2]
        flat = pixel_values.view(b * m, *pixel_values.shape[2:])
        out = self.vision(pixel_values=flat)
        tokens = out.last_hidden_state[:, 1:, :]
        d = tokens.size(-1)
        side = int(math.sqrt(tokens.size(1)))
        tokens = tokens.view(b * m, side, side, d).permute(0, 3, 1, 2)
        tokens = F.adaptive_avg_pool2d(tokens, output_size=(self.pooled_hw, self.pooled_hw))
        tokens = tokens.permute(0, 2, 3, 1).contiguous().view(b, m, self.pooled_hw * self.pooled_hw, d)
        tokens = self.proj(tokens)
        atom_mask = image_mask.unsqueeze(-1).expand(b, m, self.pooled_hw * self.pooled_hw)
        return tokens, atom_mask


class LatentSlotMNER(nn.Module):
    def __init__(
        self,
        text_name: str = "xlm-roberta-large",
        vision_name: str = "openai/clip-vit-base-patch16",
        hidden_size: int = 768,
        num_slots: int = 8,
        slot_iters: int = 3,
        pooled_hw: int = 4,
        freeze_vision: bool = True,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.text_encoder = AutoModel.from_pretrained(text_name, use_safetensors=False)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_size)
        self.image_encoder = MultiImageAtomEncoder(vision_name, hidden_size, pooled_hw=pooled_hw)
        self.boundary_head = nn.Linear(hidden_size, 3)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=hidden_size, iters=slot_iters)
        self.slot_pos_embed = nn.Parameter(torch.randn(1, num_slots, hidden_size) * 0.02)

        self.mention_proj = nn.Linear(hidden_size, hidden_size)
        self.slot_proj = nn.Linear(hidden_size, hidden_size)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.type_head = nn.Linear(hidden_size, 4)
        self.image_type_aux = nn.Linear(hidden_size, 4)

        if freeze_vision:
            for p in self.image_encoder.vision.parameters():
                p.requires_grad = False

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_proj(out.last_hidden_state)

    def build_atoms(self, text_tokens, attention_mask, image_atoms, image_atom_mask):
        b, l, h = text_tokens.shape
        _, m, p, _ = image_atoms.shape
        img_atoms = image_atoms.reshape(b, m * p, h)
        img_mask = image_atom_mask.reshape(b, m * p)
        txt_mask = attention_mask.long()
        atoms = torch.cat([text_tokens, img_atoms], dim=1)
        atom_mask = torch.cat([txt_mask, img_mask], dim=1)
        return atoms, atom_mask

    def induce_slots(self, atoms, atom_mask):
        slots, slot_to_atom_attn = self.slot_attention(atoms, atom_mask)
        slots = slots + self.slot_pos_embed[:, : slots.size(1)]
        return slots, slot_to_atom_attn

    def boundary_forward(self, text_tokens):
        return self.boundary_head(text_tokens)

    def get_span_rep(self, text_tokens, word_to_token_start, word_to_token_end, span):
        ws, we = span[:2]
        if ws not in word_to_token_start or (we - 1) not in word_to_token_end:
            return None
        ts = word_to_token_start[ws]
        te = word_to_token_end[we - 1]
        return text_tokens[ts: te + 1].mean(dim=0)

    def classify_mentions(self, span_reprs, slots):
        q = self.mention_proj(span_reprs).unsqueeze(1)
        k = self.slot_proj(slots)
        bind_logits = torch.sum(q * k, dim=-1)
        bind_attn = F.softmax(bind_logits, dim=-1)
        slot_context = torch.bmm(bind_attn.unsqueeze(1), slots).squeeze(1)
        fused = self.fuse(torch.cat([span_reprs, slot_context], dim=-1))
        type_logits = self.type_head(fused)
        return type_logits, bind_attn, slot_context, fused

    def forward(self, batch: Dict[str, Any]):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        image_mask = batch["image_mask"]

        text_tokens = self.encode_text(input_ids, attention_mask)
        boundary_logits = self.boundary_forward(text_tokens)

        image_atoms, image_atom_mask = self.image_encoder(pixel_values, image_mask)
        atoms, atom_mask = self.build_atoms(text_tokens, attention_mask, image_atoms, image_atom_mask)
        slots, slot_to_atom_attn = self.induce_slots(atoms, atom_mask)

        pooled_slots = slots.mean(dim=1)
        image_type_logits = self.image_type_aux(pooled_slots)

        span_reprs = []
        span_types = []
        owner_index = []
        owner_spans = []
        per_span_slots = []
        for b in range(input_ids.size(0)):
            for span in batch["spans"][b]:
                rep = self.get_span_rep(
                    text_tokens[b],
                    batch["word_to_token_start"][b],
                    batch["word_to_token_end"][b],
                    span,
                )
                if rep is None:
                    continue
                span_reprs.append(rep)
                span_types.append(span[2])
                owner_index.append(b)
                owner_spans.append(span)
                per_span_slots.append(slots[b])

        type_logits = None
        span_pack = None
        if span_reprs:
            span_reprs = torch.stack(span_reprs, dim=0)
            span_types = torch.tensor(span_types, device=text_tokens.device, dtype=torch.long)
            per_span_slots = torch.stack(per_span_slots, dim=0)
            type_logits, bind_attn, slot_context, fused = self.classify_mentions(span_reprs, per_span_slots)
            span_pack = {
                "span_reprs": span_reprs,
                "span_types": span_types,
                "owner_index": owner_index,
                "owner_spans": owner_spans,
                "per_span_slots": per_span_slots,
                "slot_context": slot_context,
                "fused": fused,
                "bind_attn": bind_attn,
            }

        return {
            "text_tokens": text_tokens,
            "boundary_logits": boundary_logits,
            "image_type_logits": image_type_logits,
            "atoms": atoms,
            "atom_mask": atom_mask,
            "slots": slots,
            "slot_to_atom_attn": slot_to_atom_attn,
            "span_pack": span_pack,
            "type_logits": type_logits,
        }


@dataclass
class LossConfig:
    lambda_type: float = 1.0
    lambda_aux: float = 0.2
    lambda_mask: float = 0.2
    lambda_bind_sparsity: float = 0.05
    lambda_slot_diversity: float = 0.05
    slot_masking_margin: float = 0.15


class LossComputer:
    def __init__(self, model: LatentSlotMNER, config: LossConfig):
        self.model = model
        self.config = config

    def boundary_loss(self, boundary_logits, boundary_labels):
        return F.cross_entropy(boundary_logits.view(-1, boundary_logits.size(-1)), boundary_labels.view(-1), ignore_index=-100)

    def type_loss(self, type_logits, span_pack):
        if type_logits is None or span_pack is None:
            device = type_logits.device if type_logits is not None else next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)
        return F.cross_entropy(type_logits, span_pack["span_types"])

    def image_aux_targets(self, batch, device):
        targets = []
        for spans in batch["spans"]:
            t = torch.zeros(4, dtype=torch.float)
            for _, _, ty in spans:
                t[ty] = 1.0
            targets.append(t)
        return torch.stack(targets, dim=0).to(device)

    def image_aux_loss(self, image_type_logits, batch):
        target = self.image_aux_targets(batch, image_type_logits.device)
        return F.binary_cross_entropy_with_logits(image_type_logits, target)

    def slot_masking_loss(self, span_pack, type_logits):
        if span_pack is None or type_logits is None:
            device = type_logits.device if type_logits is not None else next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)
        gold = span_pack["span_types"]
        probs = F.softmax(type_logits, dim=-1)
        gold_prob = probs.gather(1, gold.unsqueeze(1)).squeeze(1)

        attn = span_pack["bind_attn"]
        top_slot = attn.argmax(dim=-1)
        mask = torch.ones_like(attn)
        mask[torch.arange(attn.size(0), device=attn.device), top_slot] = 0.0
        masked_attn = attn * mask
        masked_attn = masked_attn / masked_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        masked_context = torch.bmm(
            masked_attn.unsqueeze(1),
            span_pack["per_span_slots"],
        ).squeeze(1)
        masked_fused = self.model.fuse(torch.cat([span_pack["span_reprs"], masked_context], dim=-1))
        masked_logits = self.model.type_head(masked_fused)
        masked_probs = F.softmax(masked_logits, dim=-1)
        masked_gold_prob = masked_probs.gather(1, gold.unsqueeze(1)).squeeze(1)
        target = torch.ones_like(gold_prob)
        return F.margin_ranking_loss(gold_prob, masked_gold_prob, target, margin=self.config.slot_masking_margin)

    def bind_sparsity_loss(self, span_pack):
        if span_pack is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        attn = span_pack["bind_attn"].clamp(min=1e-8)
        entropy = -(attn * attn.log()).sum(dim=-1).mean()
        return entropy

    def slot_diversity_loss(self, slots):
        if slots is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        norm_slots = F.normalize(slots, dim=-1)
        sim = torch.matmul(norm_slots, norm_slots.transpose(1, 2))
        b, s, _ = sim.shape
        eye = torch.eye(s, device=sim.device).unsqueeze(0)
        off_diag = sim * (1.0 - eye)
        return (off_diag ** 2).sum() / (b * s * max(s - 1, 1))

    def __call__(self, model_out, batch):
        loss_b = self.boundary_loss(model_out["boundary_logits"], batch["boundary_labels"])
        loss_t = self.type_loss(model_out["type_logits"], model_out["span_pack"])
        loss_aux = self.image_aux_loss(model_out["image_type_logits"], batch)
        loss_mask = self.slot_masking_loss(model_out["span_pack"], model_out["type_logits"])
        loss_sparse = self.bind_sparsity_loss(model_out["span_pack"])
        loss_div = self.slot_diversity_loss(model_out["slots"])
        total = (
            loss_b
            + self.config.lambda_type * loss_t
            + self.config.lambda_aux * loss_aux
            + self.config.lambda_mask * loss_mask
            + self.config.lambda_bind_sparsity * loss_sparse
            + self.config.lambda_slot_diversity * loss_div
        )
        return total, {
            "loss_boundary": float(loss_b.detach().cpu()),
            "loss_type": float(loss_t.detach().cpu()),
            "loss_aux": float(loss_aux.detach().cpu()),
            "loss_mask": float(loss_mask.detach().cpu()),
            "loss_sparse": float(loss_sparse.detach().cpu()),
            "loss_div": float(loss_div.detach().cpu()),
        }
