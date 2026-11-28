"""Skill-agnostic PyTorch Actor-Critic network with Timeline Cross-Attention and Action Masking."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from xiv_healwork_solver.rl.env import EVENT_FEAT_DIM, SKILL_FEAT_DIM, STATE_DIM


class SkillAttentionActorCritic(nn.Module):
    """Cross-attends to upcoming timeline events, scores owned skills, and feeds skill readiness to Value head."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.state_enc = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.event_enc = nn.Sequential(
            nn.Linear(EVENT_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.timeline_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)

        self.skill_enc = nn.Sequential(
            nn.Linear(SKILL_FEAT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Dedicated heads for Action 0 (WAIT) and Action 1 (FILLER_ATTACK_GCD)
        self.wait_head = nn.Linear(hidden_dim, 1)
        self.filler_head = nn.Linear(hidden_dim, 1)

        # Value head receives detached fused context + pooled legal-skill embedding so large early
        # critic TD errors cannot destabilize the BC-pretrained Actor backbone on PPO Update 1.
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_context(self, state: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        ctx = self.state_enc(state)
        ev_emb = self.event_enc(future)
        key_padding_mask = future[:, :, 11] < 0.5
        all_padded = key_padding_mask.all(dim=1)
        safe_mask = key_padding_mask.clone()
        if all_padded.any():
            safe_mask[all_padded, 0] = False

        attn_out, _ = self.timeline_attn(
            query=ctx.unsqueeze(1),
            key=ev_emb,
            value=ev_emb,
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        attn_vec = attn_out.squeeze(1)
        if all_padded.any():
            attn_vec = attn_vec.masked_fill(all_padded.unsqueeze(1), 0.0)
        return self.context_norm(ctx + attn_vec)

    def forward(
        self,
        state: torch.Tensor,
        skills: torch.Tensor,
        future: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self._encode_context(state, future)
        n_actions = skills.shape[1]
        skill_emb = self.skill_enc(skills)

        ctx_exp = ctx.unsqueeze(1).expand(-1, n_actions, -1)
        pair = torch.cat([ctx_exp, skill_emb], dim=-1)
        logits = self.action_scorer(pair).squeeze(-1)

        wait_logit = self.wait_head(ctx)
        filler_logit = self.filler_head(ctx)
        logits = torch.cat([wait_logit, filler_logit, logits[:, 2:]], dim=1)
        masked_logits = logits.masked_fill(~mask, -1e9)

        # Masked mean pool of currently legal skills -> Value head knows what resources/CDs are ready;
        # detach so early critic TD errors never corrupt the BC-pretrained Actor backbone!
        mask_f = mask.float().unsqueeze(-1)
        pooled_skills = (skill_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        value = self.value_head(torch.cat([ctx, pooled_skills], dim=-1).detach()).squeeze(-1)
        return masked_logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        skills: torch.Tensor,
        future: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state, skills, future, mask)
        if temperature != 1.0 and temperature > 0.0:
            logits = (logits / temperature).masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        if action is None:
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
