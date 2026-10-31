"""A population of small recurrent brains, evaluated as one batched tensor.

Every agent owns a slice of the weight tensors. Forward and backward run for
all agents at once with bmm. Adam is applied per slice, by hand.

The policy head is frozen inside one life. The recurrent body and the
prediction head learn to predict the next observation. There is no reward.
"""

import math

import torch
import torch.nn as nn


class BatchedBrains:
    def __init__(self, cfg):
        self.cfg = cfg
        self.N = cfg.n_slots
        self.H = cfg.hidden
        self.In = cfg.obs_dim + cfg.act_dim   # 18
        self.A = cfg.act_dim
        self.O = cfg.obs_dim

        f = torch.float32
        self.W_ih = nn.Parameter(torch.zeros(self.N, self.In, 3 * self.H, dtype=f))
        self.b_ih = nn.Parameter(torch.zeros(self.N, 3 * self.H, dtype=f))
        self.W_hh = nn.Parameter(torch.zeros(self.N, self.H, 3 * self.H, dtype=f))
        self.b_hh = nn.Parameter(torch.zeros(self.N, 3 * self.H, dtype=f))
        self.W_pred = nn.Parameter(torch.zeros(self.N, self.H, self.O, dtype=f))
        self.b_pred = nn.Parameter(torch.zeros(self.N, self.O, dtype=f))

        # policy is encoded, not learned. Frozen inside a life.
        self.W_pol = nn.Parameter(torch.zeros(self.N, self.H, self.A, dtype=f), requires_grad=False)
        self.b_pol = nn.Parameter(torch.zeros(self.N, self.A, dtype=f), requires_grad=False)

        for s in range(self.N):
            self.randomize(s)

        self.trainables = [self.W_ih, self.b_ih, self.W_hh, self.b_hh, self.W_pred, self.b_pred]
        self.m = [torch.zeros_like(t) for t in self.trainables]
        self.v = [torch.zeros_like(t) for t in self.trainables]
        self.t = 0

    # ------------------------------------------------------------------ init
    def randomize(self, s: int) -> None:
        with torch.no_grad():
            nn.init.orthogonal_(self.W_ih[s], gain=0.8)
            nn.init.orthogonal_(self.W_hh[s], gain=0.8)
            self.b_ih[s].zero_()
            self.b_hh[s].zero_()
            nn.init.normal_(self.W_pred[s], 0.0, 0.01)
            self.b_pred[s].zero_()
            nn.init.normal_(self.W_pol[s], 0.0, 0.25)
            nn.init.normal_(self.b_pol[s], 0.0, 0.05)

    def copy(self, src: int, dst: int) -> None:
        with torch.no_grad():
            for t in (self.W_ih, self.b_ih, self.W_hh, self.b_hh,
                      self.W_pred, self.b_pred, self.W_pol, self.b_pol):
                t[dst].copy_(t[src])

    def mutate(self, s: int) -> None:
        cfg = self.cfg
        with torch.no_grad():
            for t in (self.W_ih, self.b_ih, self.W_hh, self.b_hh,
                      self.W_pred, self.b_pred):
                sl = t[s]
                mask = torch.rand_like(sl) < cfg.mut_p_body
                sl.add_(mask * torch.randn_like(sl) * cfg.mut_s_body)
            for t in (self.W_pol, self.b_pol):
                sl = t[s]
                mask = torch.rand_like(sl) < cfg.mut_p_pol
                sl.add_(mask * torch.randn_like(sl) * cfg.mut_s_pol)

    # --------------------------------------------------------------- compute
    def encode(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One GRU step for every agent. x:[N,In]  h:[N,H] -> [N,H]"""
        gi = torch.bmm(x.unsqueeze(1), self.W_ih).squeeze(1) + self.b_ih
        gh = torch.bmm(h.unsqueeze(1), self.W_hh).squeeze(1) + self.b_hh
        H = self.H
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        return (1.0 - z) * n + z * h

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        h2 = self.encode(x, h)
        with torch.no_grad():
            logits = torch.bmm(h2.detach().unsqueeze(1), self.W_pol).squeeze(1) + self.b_pol
        pred = torch.bmm(h2.unsqueeze(1), self.W_pred).squeeze(1) + self.b_pred
        return h2, pred, logits

    def salience(self, x: torch.Tensor, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """|grad of chosen action logit w.r.t. input|, per element."""
        x = x.detach().clone().requires_grad_(True)
        h2 = self.encode(x, h.detach())
        logits = torch.bmm(h2.unsqueeze(1), self.W_pol).squeeze(1) + self.b_pol
        chosen = logits.gather(1, actions.view(-1, 1)).sum()
        g, = torch.autograd.grad(chosen, x, retain_graph=False)
        return g.abs().detach()

    # ------------------------------------------------------------------ adam
    def zero_grad(self) -> None:
        for t in self.trainables:
            t.grad = None

    def adam_step(self) -> None:
        cfg = self.cfg
        self.t += 1
        bc1 = 1.0 - cfg.adam_b1 ** self.t
        bc2 = 1.0 - cfg.adam_b2 ** self.t
        with torch.no_grad():
            for p, m, v in zip(self.trainables, self.m, self.v):
                if p.grad is None:
                    continue
                g = p.grad
                m.mul_(cfg.adam_b1).add_(g, alpha=1.0 - cfg.adam_b1)
                v.mul_(cfg.adam_b2).addcmul_(g, g, value=1.0 - cfg.adam_b2)
                p.add_((m / bc1) / ((v / bc2).sqrt() + cfg.adam_eps), alpha=-cfg.lr)
                p.grad = None

    # ---------------------------------------------------------------- export
    def snapshot_slice(self, s: int) -> dict:
        with torch.no_grad():
            return {
                "W_ih": self.W_ih[s].clone(),
                "b_ih": self.b_ih[s].clone(),
                "W_hh": self.W_hh[s].clone(),
                "b_hh": self.b_hh[s].clone(),
                "W_pred": self.W_pred[s].clone(),
                "b_pred": self.b_pred[s].clone(),
                "W_pol": self.W_pol[s].clone(),
                "b_pol": self.b_pol[s].clone(),
            }

    def state_dict(self) -> dict:
        with torch.no_grad():
            d = {}
            for name in ("W_ih", "b_ih", "W_hh", "b_hh", "W_pred", "b_pred", "W_pol", "b_pol"):
                d[name] = getattr(self, name).detach().clone()
            d["m"] = [t.detach().clone() for t in self.m]
            d["v"] = [t.detach().clone() for t in self.v]
            d["t"] = self.t
            return d

    def load_state_dict(self, d: dict) -> None:
        with torch.no_grad():
            for name in ("W_ih", "b_ih", "W_hh", "b_hh", "W_pred", "b_pred", "W_pol", "b_pol"):
                getattr(self, name).copy_(d[name])
            for dst, src in zip(self.m, d["m"]):
                dst.copy_(src)
            for dst, src in zip(self.v, d["v"]):
                dst.copy_(src)
            self.t = int(d.get("t", 0))
