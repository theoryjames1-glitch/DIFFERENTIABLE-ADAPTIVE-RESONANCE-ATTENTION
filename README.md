# Attention with Differentiable Adaptive Resonance (DAR)

## Abstract

Differentiable Adaptive Resonance (DAR) augments standard (self/cross) attention with a **smooth, margin-aware logit prior** that favors query–key pairs whose **cosine agreement** exceeds a **vigilance** threshold. DAR is fully differentiable, adds **no new learned tensors** (in its minimal form), preserves exact compatibility with pretrained attention stacks (set the strength to zero to recover the baseline), and can optionally use short **unrolled resonance dynamics** while remaining end-to-end trainable.

---

## 1) Base setup

Let queries, keys, values be $Q,K,V\in\mathbb{R}^{B\times H\times T_q\times d_h}$, $T_k$ for keys. Vanilla pre-softmax logits:

$$
L_{bhij}=\frac{Q_{bhi\cdot}\cdot K_{bhj\cdot}}{\sqrt{d_h}}\quad\text{for } i\in[1..T_q],\, j\in[1..T_k].
$$

Apply mask (causal or attention mask), softmax over $j$, then mix values.

---

## 2) Differentiable adaptive resonance (minimal, parameter-free)

Define cosine similarity with unit-normalized head features:

$$
\hat q_{bhi}=\frac{Q_{bhi}}{\|Q_{bhi}\|},\quad
\hat k_{bhj}=\frac{K_{bhj}}{\|K_{bhj}\|},\quad
c_{bhij}=\hat q_{bhi}\cdot \hat k_{bhj}\in[-1,1].
$$

Given **vigilance** $\rho\in[-1,1]$, **sharpness** $\alpha>0$, and **strength** $\lambda\in\mathbb{R}$, define

$$
r_{bhij}=\sigma\!\big(\alpha(c_{bhij}-\rho)\big),\qquad
L^{\text{DAR}}_{bhij}=L_{bhij}+\lambda\,r_{bhij},
$$

with $\sigma(x)=1/(1+e^{-x})$. Proceed exactly as usual: mask $L^{\text{DAR}}$, softmax, value mix.

* **Drop-in:** $\lambda=0\Rightarrow L^{\text{DAR}}=L$ (identical to baseline; load any pretrained weights).
* **Bounded prior:** $r\in(0,1)\Rightarrow \lambda r\in(0,\lambda)$ (logit perturbation is uniformly bounded).
* **Scale/rotation invariance:** cosine removes sensitivity to feature norm and orthogonal transforms per head.

---

## 3) Laws (axioms) of DAR

**A1 — Compatibility law.** Setting $\lambda=0$ recovers standard attention exactly.

**A2 — Vigilance law.** $\rho$ defines a **margin in cosine space**: $c>\rho$ pairs get a positive logit prior; $c<\rho$ pairs are suppressed, smoothly.

**A3 — Sharpness law.** $\alpha$ controls the smoothness of the margin. As $\alpha\to\infty$, $r$ approximates $\mathbf{1}\{c>\rho\}$.

**A4 — Bounded influence.** Because $r$ is bounded, DAR cannot destabilize softmax logits; it only adds a **local bias**.

**A5 — Mask invariance.** DAR is added **before masking/softmax** and respects all masks (causal, padding, cross-attention visibility).

**A6 — Temperature equivalence (local).** For two candidates $j_1,j_2$ with the same base logit gap, DAR increases their gap by $\lambda(r_{ij_1}-r_{ij_2})$, acting like a **content-adaptive temperature** that sharpens high-agreement links.

---

## 4) Gradient mechanics (sketch)

With $c=\hat q\cdot\hat k$, the needed derivatives are

$$
\frac{\partial c}{\partial q}=\frac{1}{\|q\|}\left(\hat k - c\,\hat q\right),\qquad
\frac{\partial r}{\partial c}=\alpha\,r(1-r).
$$

Let $\tilde{g}_{ij}=\partial\mathcal{L}/\partial L^{\text{DAR}}_{ij}$ be the backprop signal through the softmax. DAR adds the term

$$
\frac{\partial \mathcal{L}}{\partial q_i}\;\ni\;
\lambda\sum_j \tilde{g}_{ij}\,\frac{\partial r_{ij}}{\partial c_{ij}}\,\frac{\partial c_{ij}}{\partial q_i}.
$$

Interpretation: DAR nudges $q_i$ **toward** $\hat k_j$ if $c_{ij}$ is near but below vigilance (margin shaping), and **away** when already far above, creating self-regularized head geometry.

---

## 5) Dynamic DAR (optional unrolled resonance)

To encode fast **self-consistency**, unroll a small recurrent refinement:

$$
r^{(t+1)}_{ij}=\sigma\!\big(\alpha(c_{ij}+\beta\,r^{(t)}_{ij}-\rho)\big),\quad r^{(0)}=0,\quad t=0..T_u-1.
$$

Use $r=r^{(T_u)}$. This is fully differentiable.

**Contraction condition.** The Jacobian w\.r.t. $r$ is $\alpha\beta\,r(1-r)\le \alpha\beta/4$. If $\alpha\beta/4<1$, the map has a unique fixed point and the unroll is stable.

---

## 6) Complexity & approximations

* **Cost:** One cosine matrix + one sigmoid over $(B,H,T_q,T_k)$: $O(BHT_qT_k)$ add-on (negligible vs SDPA when $d_h$ is large).
* **Low-rank trick (optional):** Pre-project $Q,K$ to a small $r\ll d_h$ with fixed orthonormal maps (not learned) to approximate cosine cheaply; keeps the “no new learnables” principle.

---

## 7) Practical calibration

* Start with $\lambda\in[0.1,0.4]$, $\rho\in[0.4,0.7]$, $\alpha\in[4,12]$.
* If attention becomes too peaky: decrease $\lambda$ or $\alpha$, or increase $\rho$.
* For dynamic DAR: set $T_u\in\{1,2\}$, $\beta\le 0.5$, and ensure $\alpha\beta/4<1$.

---

## 8) Relationship to ART (Adaptive Resonance Theory)

DAR translates ART’s core notions into smooth, end-to-end laws:

* **Resonance:** high **cosine agreement** between query and key.
* **Vigilance:** $\rho$ defines the acceptance margin.
* **Match tracking:** the dynamic DAR refinement increases or decreases resonance based on prior resonance (via $\beta r^{(t)}$).
  Unlike classic ART, DAR uses **no discrete winners or prototype buffers**; everything is a differentiable field over all pairs.

---

## 9) Interpretability

DAR yields an explicit **resonance map** $r_{ij}$ per head:

* Visualize $r_{ij}$ alongside attention to see **why** some links win (high cosine above vigilance).
* Track vigilance-crossing statistics $\Pr[c>\rho]$ per head and layer to diagnose head selectivity.

---

## 10) Pseudocode (single head)

```python
# Q,K,V: (T_q, d_h), mask: (T_q, T_k) with False where disallowed
Qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
c  = Qn @ Kn.T                        # (T_q, T_k), cosine
r  = torch.sigmoid(alpha * (c - rho)) # (T_q, T_k)
S  = (Q @ K.T) / sqrt(d_h) + lam * r  # logits
S  = S.masked_fill(~mask, torch.finfo(S.dtype).min)
A  = softmax(S, dim=-1)
Y  = A @ V
```

---

## 11) Variants (still differentiable, simple to swap)

* **Centered gate:** use $c'=\tfrac{1}{2}(c+1)\in[0,1]$, gate via $\sigma(\alpha(c'-\rho'))$.
* **Tanh gate:** $r=\tfrac{1}{2}\big(1+\tanh(\alpha(c-\rho))\big)$ for gentler tails.
* **Piecewise-linear gate:** $r=\mathrm{clip}(\gamma(c-\rho)+\tfrac12,0,1)$ to bound gradient magnitude.
* **Content-adaptive λ:** make $\lambda$ a deterministic function of signal quality (e.g., norm of Q/K); still no new parameters.

---

## 12) What DAR buys you

* **Drop-in to any attention** (self, cross, MQA, multi-query): zero retraining friction.
* **Margin-aware routing** that steers attention using cosine agreement without altering masks or value mixing.
* **Stable & bounded** bias that is easy to calibrate and debug.
* **Exact backward compatibility** with pretrained weights ($\lambda=0$).

---

### TL;DR

Attention with Differentiable Adaptive Resonance = **standard scaled dot-product attention** plus a **bounded, smooth, cosine-selective logit prior** $\lambda\,\sigma(\alpha(c-\rho))$. It’s **fully differentiable**, **parameter-free** (in the minimal form), **stable** under mild conditions when made dynamic, and **compatible** with existing pretrained models while offering interpretable, margin-based control of attention selectivity.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class DARAttention(nn.Module):
    """
    GPT-2 style multi-head attention with a parameter-free, differentiable
    adaptive resonance (DAR) bias added to the logits.

        cos_ij = cosine(q_i, k_j)
        r_ij   = sigmoid(alpha * (cos_ij - rho))
        logits += lam * r_ij

    - Set lam=0 for vanilla attention.
    - Adds no new learnable tensors beyond standard GPT-2 (c_attn/c_proj).
    - Uses dtype-safe masking for stability in fp16/bf16.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias_qkv: bool = True,
        bias_proj: bool = True,
        lam: float = 0.3,
        rho: float = 0.6,
        alpha: float = 8.0,
        iters: int = 0,
        beta: float = 0.5,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.dh = dim // n_heads

        # Match GPT-2 shapes (bias=True to align with common checkpoints)
        self.c_attn = nn.Linear(dim, 3 * dim, bias=bias_qkv)
        self.c_proj = nn.Linear(dim, dim, bias=bias_proj)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        # DAR hyperparams (not in state_dict)
        self.lam = float(lam)      # resonance strength
        self.rho = float(rho)      # vigilance threshold (cosine)
        self.alpha = float(alpha)  # gate sharpness
        self.iters = int(iters)    # unrolled refinement steps (0..2 recommended)
        self.beta = float(beta)    # recurrence weight if iters>0

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.ones(T, T, device=device, dtype=torch.bool).tril_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.dh).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))  # (B,H,T,dh)

        # Base scaled dot-product logits
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.dh)  # (B,H,T,T)

        # Differentiable adaptive resonance bias (parameter-free)
        if self.lam != 0.0:
            eps = 1e-8
            qn = q / (q.norm(dim=-1, keepdim=True) + eps)
            kn = k / (k.norm(dim=-1, keepdim=True) + eps)
            cos = torch.einsum("bhid,bhjd->bhij", qn, kn)  # cosine in [-1,1]
            if self.iters > 0:
                r = torch.zeros_like(cos)
                for _ in range(self.iters):
                    r = torch.sigmoid(self.alpha * (cos + self.beta * r - self.rho))
            else:
                r = torch.sigmoid(self.alpha * (cos - self.rho))
            scores = scores + self.lam * r

        # Causal mask (dtype-safe)
        mask = self._causal_mask(T, x.device).view(1, 1, T, T)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # Attention + output
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)  # (B,H,T,dh)

        # Merge heads
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.dim)
        out = self.c_proj(ctx)
        out = self.resid_drop(out)
        return out


class GPT2_ARA(nn.Module):
    """
    Minimal GPT-2 stack with DARAttention blocks.

    Args:
        vocab_size: tokenizer vocab size.
        dim: model width.
        depth: number of transformer blocks.
        n_heads: number of attention heads.
        max_pos: maximum sequence length for positional embeddings.
        lam, rho, alpha, iters, beta: DAR hyperparameters (see DARAttention).
        attn_dropout, resid_dropout: dropout probabilities.
        bias_qkv, bias_proj: include bias on qkv/proj (True to match GPT-2).
        weight_tying: tie lm_head.weight to token embedding weights.
    """
    def __init__(
        self,
        vocab_size: int,
        *,
        dim: int = 256,
        depth: int = 4,
        n_heads: int = 4,
        max_pos: int = 2048,
        lam: float = 0.3,
        rho: float = 0.6,
        alpha: float = 8.0,
        iters: int = 0,
        beta: float = 0.5,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias_qkv: bool = True,
        bias_proj: bool = True,
        weight_tying: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.n_heads = n_heads

        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_pos, dim)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(dim, eps=1e-5),
                "attn": DARAttention(
                    dim, n_heads,
                    attn_dropout=attn_dropout, resid_dropout=resid_dropout,
                    bias_qkv=bias_qkv, bias_proj=bias_proj,
                    lam=lam, rho=rho, alpha=alpha, iters=iters, beta=beta
                ),
                "ln2": nn.LayerNorm(dim, eps=1e-5),
                "mlp": nn.Sequential(
                    nn.Linear(dim, 4 * dim, bias=True),
                    nn.GELU(),
                    nn.Linear(4 * dim, dim, bias=True),
                ),
            }) for _ in range(depth)
        ])

        self.ln_f = nn.LayerNorm(dim, eps=1e-5)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if weight_tying:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Strict validation to catch common pitfalls early
        if not torch.is_tensor(input_ids):
            raise ValueError("input_ids must be a torch.Tensor.")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # (1, T)
        if input_ids.dim() != 2:
            raise ValueError("Expected input with 2 dims (B, T).")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be dtype torch.long (token ids).")

        B, T = input_ids.shape
        if T >= self.pos_emb.num_embeddings:
            raise ValueError("Sequence length exceeds positional embedding size.")

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        h = self.embed(input_ids) + self.pos_emb(pos)

        for blk in self.blocks:
            h = h + blk["attn"](blk["ln1"](h))
            h = h + blk["mlp"](blk["ln2"](h))

        h = self.ln_f(h)
        return self.lm_head(h)  # (B, T, vocab_size)
```
