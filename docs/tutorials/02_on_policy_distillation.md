# Tutorial 02: Why On-Policy Distillation Works

Tutorial 01 showed you the **mechanics**: rollout → teacher → train → sync. This tutorial answers the *why*. We derive the reverse-KL loss the actor optimizes, explain why we sample from the student instead of from a static dataset, and connect the design to PPO/GRPO so the planned `opd_rl` mode in M3 reads as a one-line change.

The technical content paraphrases the Thinking Machines post *On-Policy Distillation* (TM 2025; see References). Citations point at primary sources where possible.

> **TL;DR.** On-policy distillation = sample from the student, score with a frozen teacher, update the student with reverse-KL. You get RL's correct credit-assignment (the student practices on states it actually visits) **and** dense supervision (one signal per token instead of one scalar per episode). TM 2025 reports ~10× lower GPU cost than RL at higher AIME'24 on Qwen3-8B.

## 1. Two failure modes of off-policy SFT

Off-policy distillation = supervised fine-tuning (SFT) on teacher-generated text. The student is trained to match teacher tokens on **teacher-visited states**. Two well-known problems follow:

1. **Compounding error / exposure bias.** At training time the student sees only states the teacher would have reached. At inference the student samples its own next token; a single early mistake puts it in a state the teacher never visits, where the student's next-token distribution has never been supervised. Errors compound autoregressively. This is the same gap that motivated **DAGGER** (Ross et al. 2010): supervise the policy on its *own* state distribution, not the expert's.
2. **Style without substance.** Off-policy SFT teaches the student to imitate the teacher's surface form (tone, length, formatting) without enforcing factual or step-level correctness — there is no signal that says "this token, given the trajectory you actually produced, is wrong" [TM 2025].

A chess analogy (TM 2025): SFT is *watching* a grandmaster play; on-policy distillation is *playing yourself* while a grandmaster grades each of your moves from "blunder" to "brilliant."

## 2. Why dense beats sparse: the information-theoretic argument

Standard RL with a terminal reward (RLHF, GRPO, verifier-only) provides $O(1)$ bits per episode — one scalar at the end. The policy gradient has to spread that scalar across all $N$ tokens via the return-to-go.

A distillation signal scored against a teacher provides $O(N)$ bits per episode — one signal per token. For a fixed compute budget, the per-token signal-to-noise ratio is dramatically higher [TM 2025]. This is the load-bearing argument for the **dense teacher logprob service** in this repo (`opd/teacher/eager.py`): the teacher is doing the work of a process reward model (cf. Lightman et al. 2023, *Let's Verify Step by Step*) for free, at every token, without a separate trained PRM.

## 3. Deriving the loss

### 3.1 Reverse KL on autoregressive policies

Both student $\pi_S$ and teacher $\pi_T$ are autoregressive distributions over token sequences. For a sequence $x = (x_1, \dots, x_N)$,

\[
\pi(x) = \prod_{t=1}^{N} \pi(x_t \mid x_{<t}).
\]

The **reverse** KL divergence, with the expectation under the student, is

\[
\mathrm{KL}\bigl(\pi_S \,\|\, \pi_T\bigr)
= \mathbb{E}_{x \sim \pi_S}\!\left[ \log \pi_S(x) - \log \pi_T(x) \right].
\]

Substituting the autoregressive factorization and pulling the sum out of the expectation,

\[
\mathrm{KL}\bigl(\pi_S \,\|\, \pi_T\bigr)
= \mathbb{E}_{x \sim \pi_S} \sum_{t=1}^{N} \bigl[ \log \pi_S(x_t \mid x_{<t}) - \log \pi_T(x_t \mid x_{<t}) \bigr].
\]

The per-token term

\[
\mathcal{L}_{\mathrm{KL}, t} \;=\; \log \pi_S(y_t \mid s_t) \;-\; \log \pi_T(y_t \mid s_t)
\]

is exactly what `opd/loss/kl.py:reverse_kl_loss` computes per position before masking and averaging. (Also matches `docs/superpowers/specs/2026-05-26-opd-compute-infra-design.md` line 258.)

### 3.2 Why *reverse* and not *forward* KL

Forward KL — $\mathrm{KL}(\pi_T \| \pi_S)$, expectation under the teacher — would push the student to **cover** every mode the teacher places probability on. With a wide teacher this is wasteful: most of the teacher's distribution is irrelevant, and the student's limited capacity gets diluted.

Reverse KL is **mode-seeking**: the expectation is under the *student*, so we only pay for distributions over states the student actually produces. The student learns to put mass *where the teacher already does*, on its own state distribution. This matches the mode-seeking vs. mean-seeking discussion in Gu et al. 2023 (*MiniLLM*) and is what makes on-policy distillation a sharpening procedure rather than a covering one.

### 3.3 Discount factor zero

The TM recipe sets the per-token discount factor to zero: each token is scored only by its own KL term, not by a return-to-go over future tokens. The argument is that the teacher's logprob at position $t$ already encodes everything we want to say about token $y_t$ in state $s_t$ — there is no value bootstrapping, no GAE, no critic [TM 2025]. This is why **OPD's actor needs no value head** and why `StudentActor.train_step` in this repo is ~10 lines.

## 4. The RL connection (foreshadows M3 `opd_rl`)

Reverse-KL distillation can be rewritten as a degenerate policy gradient. Define the per-token **advantage**

\[
A_t \;=\; -\bigl( \log \pi_S(y_t \mid s_t) - \log \pi_T(y_t \mid s_t) \bigr).
\]

Then a clipped importance-sampling objective (PPO-style) over student rollouts

\[
\mathcal{J}_{\mathrm{PPO}}(\theta) \;=\; \mathbb{E}_t \!\left[ \min\!\Bigl(r_t(\theta)\,A_t,\ \mathrm{clip}\bigl(r_t(\theta),\,1{-}\epsilon,\,1{+}\epsilon\bigr)\,A_t\Bigr) \right],
\qquad
r_t(\theta) = \frac{\pi_\theta(y_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(y_t|s_t)}
\]

reduces to on-policy distillation when $A_t$ is the negative reverse KL and `epochs_per_step = 1` (so $r_t \equiv 1$ and the clip is inactive). With more epochs and clipping, you get a PPO-flavored variant; with group-relative normalization across multiple rollouts per prompt you get GRPO. This is the spec at lines 251–269 and the planned `loss.mode=opd_rl` for M3 — implementationally a one-line change on top of an RL trainer that already does clipped IS [TM 2025].

Practical consequence: **a working OPD trainer ships you 80% of the way to PPO/GRPO**. The reference, reward, and critic modules from RLHF stay deleted; only the IS-and-clip wrapper around the existing KL term is added. This is the route from `opd/loss/kl.py` (M1) to `opd_rl` (M3).

## 5. Algorithm → code

The TM recipe maps onto this repo's `LocalDriver.run` (`opd/runtime/local_driver.py:99`) exactly:

| TM step                                                                       | This repo                                                                     | Tensor / shape note                                                                              |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Sample $x \sim \pi_S$ given prompts                                           | `StudentRollout.generate` in `opd/rollout/eager.py`                           | `token_ids: [B, T]`, `student_logprobs: [B, T]` over the *sampled* token at each step            |
| Compute $\log \pi_T(x_t \mid s_t)$ on the same tokens                         | `TeacherScorer.score` in `opd/teacher/eager.py`                               | `teacher_logprobs: [B, T]` — gathered on sampled ids; the `teacher_signal=sampled` default       |
| Form $\mathcal{L}_{\mathrm{KL},t} = \log \pi_S - \log \pi_T$ and reduce       | `reverse_kl_loss` in `opd/loss/kl.py`                                         | masked mean over response positions; the prompt prefix is excluded via `attention_mask`          |
| One optimizer step                                                            | `StudentActor.train_step` in `opd/actor/eager.py`                             | AdamW, no clipping, no value head                                                                |
| Refresh the rollout policy (`weight sync`)                                    | `self.rollout.model.load_state_dict(self.actor.model.state_dict())`           | Tiny tier: in-process copy; `sync_bytes` is recorded so larger tiers can show FSDP→vLLM cost      |
| Persist per-token signal for debugging                                        | `_build_token_samples` in `opd/runtime/local_driver.py`                       | Top-N sequences by $\max_t |\mathcal{L}_{\mathrm{KL},t}|$ — see §6 for the magnitude convention   |

At the `tiny` tier all five steps run sequentially in one Python process; at `qwen3_small` (M2) the same calls fan out across Ray placement groups. The control flow is identical.

## 6. Per-token KL visualization

The explorer's **Token KL** panel (`explorer/src/opd/TokenHeatmap.tsx`) renders a few sampled rollouts with each token shaded by the same $\mathcal{L}_{\mathrm{KL},t}$ the loss computes. This is the same kind of figure TM 2025 uses to debug a SimpleBench physics trace.

Conventions in this repo:

- **Color intensity = magnitude, not signed value.** The shading alpha is
  \[
  \alpha_t \;=\; \min\!\left(1,\ \frac{|\mathcal{L}_{\mathrm{KL},t}|}{P_{95}(|\mathcal{L}_{\mathrm{KL}}|)}\right),
  \]
  capped at the 95th percentile of $|\mathcal{L}_{\mathrm{KL}}|$ either per-step or globally (toggleable in the UI). Magnitude — not signed value — because $\log \pi_S - \log \pi_T$ is mixed-sign early in training and a debug view that hides the negative half buries half the signal (see the code review on this PR for the bug that motivated the convention).
- **Sample selection ranks by magnitude.** `_build_token_samples` picks the top-$N$ sequences by `per_token_kl.abs().amax(dim=1)` so a strongly negative spike is just as eligible as a strongly positive one. Default $N=4$ via `log_token_samples` in `configs/tier_tiny.yaml`.
- **Hover for the exact value.** The tooltip on each token shows the signed reverse-KL; the direction matters for debugging (e.g. negative = student more confident than teacher = possible overfit to local prompt features).

What you should *look* for, per TM 2025:

- **Bright tokens that start a phrase.** TM's example: when the student is about to commit to a wrong reasoning step, the first token of the wrong phrase lights up — the teacher's distribution put very little mass on it. The wrong *answer* token several positions later often does **not** light up: by then the answer is fully determined by the preceding sequence and the teacher would predict the same continuation.
- **Quiet stretches with one outlier.** Indicate a generally on-distribution rollout where the student made one specific bad choice. These are the highest-value debug cases.
- **Uniformly bright sequences.** Usually mean the student and teacher diverge in style or formatting rather than in any one decision. Less informative.

## 7. Empirical results (paraphrased from TM 2025)

These numbers are reproduced for orientation only — this repo does not yet run Qwen3-8B (M2 milestone). All from TM 2025 unless noted:

| Setting                                            | AIME'24 | GPU·h     | Notes                                                                                     |
| -------------------------------------------------- | ------- | --------- | ----------------------------------------------------------------------------------------- |
| Qwen3-8B-Base + SFT-400K on OpenThoughts-3         | 60%     | —         | Initialization. Extrapolates to ~70% at 2M prompts.                                       |
| Qwen3-8B-Base + RL                                 | 67.6%   | 17,920    | Qwen Team 2025, Table 21.                                                                 |
| Qwen3-8B-Base + SFT-400K → on-policy distillation  | **74.4%** | **~1,800** | TM 2025. ~10× cheaper than the RL baseline at higher score.                              |
| Self-distillation (RL → distill from RL model)     | ~70%    | —         | 7–10× fewer steps than re-running RL.                                                     |
| Personalization midtrain → distill (IF-eval)       | 83%     | —         | Recovery from 45% post-midtrain — the regression IF-eval suffers from raw SFT on outputs. |
| LoRA (r=32) post-distillation                      | within 6% of full FT | — | Compare to ~13% gap post-SFT — distillation makes LoRA carry further.                     |

These are TM's numbers and the methods page in their post should be the citation in any downstream write-up.

## 8. What this repo does *not* do (yet)

The spec at `docs/superpowers/specs/2026-05-26-opd-compute-infra-design.md` lays out the staging:

- **Top-k sparse KL** (`teacher_signal=topk`, M2). Today the loss gathers teacher logprobs only on the sampled token — fine for the `tiny` tier but loses signal at scale. Top-k mode would let the teacher contribute a distribution over its top-$k$ alternatives per position.
- **`opd_rl` loss mode** (M3). The IS-and-clip wrapper from §4. The data path (`TrainBatch.old_logprobs`) is already plumbed.
- **LoRA adapters.** TM reports LoRA-r32 nearly matches full fine-tuning after distillation; this repo has no LoRA path yet.
- **Real models.** `tiny` uses `SyntheticQwen3` (`opd/models/synthetic_qwen3.py`). M2 plugs in HF Qwen3 via vLLM rollout + FSDP actor.

## References

- **Agarwal et al. (2023).** *On-Policy Distillation of Language Models.* The recipe of sampling from the student and using teacher feedback per token. The TM post extends this work with the RL framing and modern model scale.
- **Gu et al. (2023).** *MiniLLM: Knowledge Distillation of Large Language Models.* Reverse-KL with on-policy samples; mode-seeking vs. mean-seeking analysis.
- **Qwen Team (2025).** *Qwen3 Technical Report.* Source of the RL baseline (Table 21) and the OpenThoughts-3 SFT setup.
- **Lightman et al. (2023).** *Let's Verify Step by Step.* Process reward modeling — the closest sparse-RL analog to what a teacher provides for free in OPD.
- **Ross et al. (2010).** *DAGGER.* The canonical exposure-bias / on-policy correction in imitation learning.
- **Rafailov et al. (2023).** *Direct Preference Optimization (DPO).* Tangential; cited by TM as a non-RL alternative to RLHF.
- **Wang et al. (2025).** *Beyond the 80/20 Rule.* Forking-token / decision-token analysis relevant to interpreting bright tokens in the per-token KL view.
- **Thinking Machines (2025).** *On-Policy Distillation.* https://thinkingmachines.ai/blog/on-policy-distillation — the primary source for this tutorial's framing, empirical numbers, and visualization conventions.

## Next steps in this repo

- **Tutorial 03** (planned): trace `sync_bytes` and weight-sync cost at scale; FSDP gather → vLLM `load_weights`.
- **Tutorial 04** (planned): `loss.mode=opd_rl` head-to-head with `kl`.
- **Spec & milestones:** [docs/superpowers/specs/2026-05-26-opd-compute-infra-design.md](../superpowers/specs/2026-05-26-opd-compute-infra-design.md).
