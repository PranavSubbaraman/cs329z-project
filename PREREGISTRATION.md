# Preregistration

Written 2026-11-06, before any baseline, static-review or agent run on the benchmark. These rules are not
changed after results are seen; any later deviation is listed under "Deviations" with its reason.

## Benchmark
9 instances from linkedin/Liger-Kernel (`benchmark/instances.json`). Labels are in `harness_only/labels.json`.

| pair | class | buggy | fixed |
|---|---|---|---|
| P1 TVD (#1373) | logic | 39adfeef | 559d33e7 |
| P2 FLCE column strides (#1500) | index arithmetic | 395d6166 | ae47958e |
| P3 FLCE reduction="none" (#1298) | gradient formula | e44a5fe8 | 9ac26bb8 |
| P4 FLJSD bf16 logits (#1432) | precision | 5b828eef | ed1f856d |
| B5 CE ignore_index rows (#1594, open) | side effect | 8c7a5078 | (none: buggy-only) |

Each run sees one instance: the exported tree (no .git), the kernel's `task.md`, `entry.py` and `reference.py`.
Execution happens on one NVIDIA RTX A4000 (sm_86), torch 2.15.0+cu132, triton 3.9.0. LLM:
`anthropic/claude-sonnet-5.5` through OpenRouter, provider pinned to Anthropic.

## Conditions
1. **Repo tests**: Liger's own test files for the kernel at the instance's commit (recorded in Phase 1). A failing
   test is a bug claim; all passing is no_bug.
2. **Random differential testing**: specs drawn by `oracle.spaces.sample` from the kernel's documented domain.
   Each spec runs `grad_compare` and every property check (`torch.compile` on every 10th spec only, for cost).
   It stops at the first violation, which becomes the counterexample, or when the budget runs out (no_bug).
   Evaluation seeds start at 0; the calibration seeds (10000+) are disjoint.
3. **Static review**: the same model reads the tree, task files and reference, never executes anything, and
   returns `{verdict, counterexample (input spec), file:line, evidence}`.
4. **Agent without z3_check**: the full agent loop with the `z3_check` tool removed.
5. **Full agent**: ReAct loop with list_files, read_file, run_gradient_test, check_properties, z3_check,
   conformance_check, replay and submit_verdict.

Every LLM-based condition (3, 4, 5) runs **3 times per instance**. Results are reported as mean and range.

## Budgets
- Agent (conditions 4 and 5): at most **25 tool calls** and **10 minutes wall-clock** per run. Hitting either limit
  forces an `inconclusive` verdict unless one was already submitted.
- Static review: one model call, with up to 2 retries only for malformed output.
- Random testing: for each agent run, one random-testing run with the **same GPU-seconds** as that agent run used
  (3 runs per instance, seeds 0, 1000, 2000). GPU-seconds = wall-clock time spent inside sandboxed executions that
  run the kernel (run_gradient_test, check_properties, conformance_check, replay; process start-up included).
  Z3 time is CPU-only and logged separately.
- **Preliminary (midway) random run, labeled as such:** 200 specs per instance (seeds 0–199), not matched to any
  agent budget. It is superseded by the budget-matched run.

## Scoring (identical for every method)
Every bug claim is replayed by the harness with the trusted oracle (`oracle.replay`) on the instance's own tree.
A claim **replays** iff replay reports at least one violation: a failed `grad_compare` or a failed property.
A kernel crash alone is not a violation.
- Buggy instance: a replayed claim is a **detection**. An unconfirmed claim is a **miss**, with a note. A no_bug or
  inconclusive verdict is a **miss**.
- Fixed instance:
  - an unconfirmed claim is a **false alarm**;
  - a replayed claim is **flagged as a possible new bug**, checked against `known_other_bugs` and reported
    separately;
  - no_bug is **correct**;
  - inconclusive is reported separately and is neither correct nor a false alarm.
- **Proof**: verdict no_bug with proof_status UNSAT, where conformance passed, on a fixed instance.
- **False proof**: the same on a buggy instance.
- **Localization**: the reported `file:line` is inside the fix diff's changed lines (old-side hunk lines; pure
  insertions use their two anchor lines). For B5 the issue-derived lines are `ops/cross_entropy.py:104-108`.
  Compared harness-side only.
- Also recorded per run: tokens (prompt, completion, cached), cost (USD from OpenRouter), wall-clock, GPU-seconds,
  tool calls, and the serving provider.

## Metrics table
`results/midway_table.{csv,md}`: methods × {detection rate on buggy instances, false-alarm rate on fixed instances,
proof rate, false-proof rate, localization rate, cost, wall-clock, GPU-seconds}. Broken down by bug class. Mean and
range over the 3 runs where applicable.

## Oracle tolerance (frozen)
`oracle/tolerances.py`:
`max|T-R| <= c_out*u(d)*max|R| + c_acc*u32*log2(2n)*max|R| + K*max|R32-R| + c_out*eta(d)*sqrt(n)` (the last term added before any baseline result was used; see Deviations), with **c_out = 4, c_acc = 4, K = 8**.
The constants were chosen before calibration. The calibration on fixed commits (seeds 10000+, 100 specs each) is
used only to confirm zero false alarms. If it had shown false alarms, the constants or the reference would have
been revised before any baseline run, and the revision recorded here.

## Deviations
1. **2026-11-06, tolerance rule: an underflow term was added after calibration, before any baseline result was used.**
   - Calibration on `fljsd@ed1f856d` (fixed) gave 2 of 100 float16 `grad_student_input` false alarms (ratios 1.2 and
     3.8).
   - Both are fp16 underflow. Gradients of about 1e-5 to 1e-7 are subnormal in fp16. In one case the kernel returns
     0 where the reference is 1.85e-7, about 3 subnormal steps.
   - Added term (4): `c_out*eta(d)*sqrt(n)`, the absolute part of the IEEE rounding model. It is negligible for
     float32 and bfloat16. The constants are unchanged, and the calibration on every fixed commit is re-run with the
     revised rule.
2. **2026-11-06, process: the first preliminary random-testing run started before calibration finished.** That run
   used the unrevised rule. It is discarded unread and re-run with the revised rule. Its raw output is kept at
   `results/prelim_random_discarded/` for transparency.
3. **2026-11-07, tolerance rule: a floor for identically-zero outputs, after the preliminary random run (midway) hit
   a false alarm.**
   - On `flce@ae47958e` (fixed), spec 27 has V = 1. The exact loss is identically 0, and the float32 reference
     also computes exactly 0, so term (3) allowed no error. The kernel returns 3.8e-7, float32 rounding of logits
     of magnitude about 4.
   - Fix: only for an output whose float64 and float32 references are both exactly 0, allow
     `c_acc*u32*log2(2n)*max(1, max|input|)`. Gradients are unchanged.
   - Calibration and the preliminary random run are re-run with the revised rule. The run that hit this is kept at
     `results/prelim_random_v2_pre_zero_floor/`.
   - The preliminary run is not a scored baseline; the budget-matched final run uses the revised rule throughout.
4. **2026-11-07, flce domain: V >= 2 (a single class is out of scope), after the re-run preliminary random run.**
   - On `flce@ae47958e` (fixed), spec 195 has V = 1. With one class, cross entropy's loss and *all gradients* are
     identically 0 for every input. The kernel returns float32 cancellation residue of about 1e-7, and the
     gradient comparison is strict at exact zeros by design, which is what lets exactly-zero gradients at TVD ties
     count.
   - Rather than weaken that, flce's documented domain now requires V >= 2, matching ce's `task.md`, and the
     sampler draws V from [2, 512].
   - All four flce preliminary random runs are redone. The superseded results are in
     `results/prelim_random_v3_flce_v1/`.
   - The agent and static-review runs, made with the previous `task.md` wording, are unaffected: none used V = 1.
