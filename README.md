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
