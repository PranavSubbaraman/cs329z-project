# Finding silent gradient bugs in Triton training kernels

CS 329Z project. An LLM agent reads a fused Triton kernel's source (from [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)), guesses which input condition could break its hand-written backward pass, builds inputs that trigger it, and checks the gradients against a float64 PyTorch reference. It can also use Z3 to search for counterexamples or prove equivalence. Bug claims count only if the harness can replay them.

The benchmark is 5 real Liger-Kernel bugs (logic, index arithmetic, gradient formula, precision, side effect), as 9 buggy/fixed instances. The agent is compared against Liger's own tests, random differential testing, and a static LLM review that can read the code but not run it.

Each step below was tested on a fresh clone. Times and costs are from those runs.

---

## 1. Requirements

| | Laptop, no GPU (quick start) | Linux + NVIDIA GPU (reported results) |
|---|---|---|
| OS | macOS or Linux | Linux |
| Python | 3.12 | 3.12 |
| GPU | none: kernels run on the Triton CPU interpreter (`TRITON_INTERPRET=1`) | CUDA GPU, compute capability ≥ 8.0 (results used an RTX A4000) |
| Driver | – | one that supports CUDA 13 (torch 2.15.0 wheels bundle CUDA 13.2; results used driver 610) |
| Disk | ~3 GB (+ ~7 GB to build Triton on macOS) | ~6 GB |

All instances run on the CPU interpreter except the precision pair (`fljsd`). That pair's fix only takes effect on CUDA with compute capability ≥ 8, so on CPU the buggy and fixed versions behave identically.

## 2. Install

```bash
git clone https://github.com/PranavSubbaraman/cs329z-project.git
cd cs329z-project            # results are on the `dev` branch: git checkout dev
```

**Linux with a CUDA GPU:** pip pulls the matching Triton wheel.
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**macOS, or any machine without CUDA:** PyPI has no macOS Triton wheel, so build Triton 3.9.0 from source. The interpreter still needs `libtriton`.
```bash
uv venv --seed --python 3.12 .venv          # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
./scripts/install_triton_macos.sh            # clones + builds triton v3.9.0 into .deps/ (~15-25 min, ~7 GB)
```

Check the install:
```bash
.venv/bin/python -m pytest tests -q          # 582 reference math checks, ~5 s
TRITON_INTERPRET=1 .venv/bin/python scripts/env_report.py   # or omit TRITON_INTERPRET on a GPU
```

## 3. API key (OpenRouter)

The agent calls Claude through OpenRouter's OpenAI-compatible API. Create a key at <https://openrouter.ai/keys>, then use either option below. `.env` is git-ignored.

```bash
cp .env.example .env        # then edit .env:  OPENROUTER_API_KEY=sk-or-v1-...
# or
export OPENROUTER_API_KEY=sk-or-v1-...
```

The defaults, all overridable in `.env` or the environment:
- `KV_MODEL=anthropic/claude-sonnet-5.5`
- `KV_PROVIDER=anthropic` (provider pinned, no fallback)
- `KV_MAX_TOKENS=8192`

Every call records the serving provider, tokens and cost (`agent/llm.py`). `python scripts/smoke_openrouter.py` makes one $0.002 test call with a dummy tool.

## 4. Get the benchmark sources

The agent sees an exported Liger-Kernel source tree (no `.git`, so no history or commit messages) at one commit. This downloads those trees from GitHub into `work/trees/`, about 40 MB per commit:

```bash
.venv/bin/python scripts/export_trees.py               # all 9 instances (~1 min)
.venv/bin/python scripts/export_trees.py --kernel tvd  # just the TVD pair (quick start)
```

## 5. Quick start: run the agent on one task

```bash
.venv/bin/python scripts/run_agent.py --instance tvd@39adfeef
```

This runs the full agent on the TVD loss at a buggy commit:
1. It builds a sandbox under `work/sandboxes/`.
2. The agent reads the kernel, tests its hypothesis, and submits a verdict.
3. The harness replays the agent's counterexample with the trusted oracle and scores it.

On a laptop CPU this took 16–30 s and $0.07–0.19.

Expected output (trimmed):
```
"verdict": "bug",
"location": "liger/src/liger_kernel/ops/tvd.py:89",   <- the exact line the real fix changed
"score": "detection",
"localized": true,
```

Every run appends one scored line to `results/runs.jsonl` and saves its full trace to `results/traces/<run_id>.json`. To read a trace:
```bash
.venv/bin/python scripts/show_trace.py results/traces/<run_id>.json
```

More examples. Instances are listed in `benchmark/instances.json`; `--condition` is `full`, `no_z3` (agent without Z3) or `static` (reads code only, never executes):
```bash
.venv/bin/python scripts/run_agent.py --instance tvd@559d33e7               # fixed commit; expect "correct_no_bug"
.venv/bin/python scripts/run_agent.py --instance ce@8c7a5078 --condition static
```

**No API key?** `./demo_midway.sh` reproduces the headline TVD result in about 20 s. It runs Liger's recorded test result, random testing, and a live replay of the agent's recorded counterexample on both commits. It needs `export_trees.py --kernel tvd` first.

## 6. Reproduce the evaluation

The numbers in the report come from the committed files in `results/`. To rebuild the results table from them:
```bash
.venv/bin/python scripts/summarize_results.py     # -> results/midway_table.{md,csv}
```

To re-run everything from scratch on a Linux GPU machine, use these scripts from the repo root. They use `.venv-gpu/` (`scripts/gpu/setup_env_linux.sh` creates it). Each writes to `results/`.

| What | Command | Time / cost |
|---|---|---|
| Benchmark validation: repro per pair on buggy and fixed, Liger's own test suites, reference-vs-kernel agreement, network check | `scripts/gpu/phase1_gpu.sh` | ~10 min |
| Oracle checks: targeted input per bug, plus calibration with 100 random specs per fixed commit | `scripts/gpu/phase2_gpu.sh 100` | ~10 min |
| Random differential testing (≤200 specs per instance, seed 0) | `scripts/gpu/prelim_random_gpu.sh` | ~10 min |
| Agent + static review on all 9 instances (18 runs, 3 in parallel) | `scripts/gpu/midway_batch.sh agents` | ~30 min, ~$11 |
| One method on one instance | `.venv-gpu/bin/python scripts/run_agent.py --instance <id> --condition full\|no_z3\|static --runs 3` | ~$0.6/run |
| Random testing on one instance | `KV_KERNEL=<k> KV_TASK_DIR=benchmark/kernels/<k> scripts/harness/in_tree.sh <commit> .venv-gpu/bin/python baselines/random_testing.py --seed 0 --max-specs 200 --out <file>` | 10–220 GPU-s |
| Rebuild the table | `python scripts/summarize_results.py` | instant |

`scripts/harness/in_tree.sh <commit> <cmd>` runs any command against one exported tree. It sets `PYTHONPATH`, and sets `TRITON_INTERPRET=1` unless you pass `TRITON_INTERPRET=0` on a GPU.

The metrics, scoring rules and budgets were fixed in advance in **`PREREGISTRATION.md`**, including every deviation and why it was made. On a buggy instance, a bug claim counts as a detection only if the harness replays its counterexample. On a fixed instance, an unconfirmed claim is a false alarm.

**On a Slurm cluster** (how the reported runs were made, on Stanford SC): allocate a GPU, then run the scripts inside the allocation.
```bash
salloc --no-shell --partition=<gpu-partition> --account=<account> --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=04:00:00
srun --jobid=<jobid> --overlap scripts/gpu/setup_env_linux.sh        # once
srun --jobid=<jobid> --overlap scripts/gpu/phase1_gpu.sh             # etc.
```
The agent needs outbound HTTPS to `openrouter.ai` from the compute node; `phase1_gpu.sh` checks this. Put the key in `.env` with `chmod 600`.

## 7. Repository layout

| Path | What it is |
|---|---|
| `agent/` | `llm.py` (OpenRouter client), `sandbox.py` (per-run sandbox + guard), `tools.py` (the agent's 8 tools), `prompts.py`, `loop.py` (ReAct loop, budgets: 25 tool calls / 10 min), `score.py` (harness-side scoring) |
| `oracle/` | `compare.py` (`grad_compare`: kernel in native dtype vs. reference in float64), `tolerances.py` (the tolerance rule, documented), `properties.py` (zero-upstream, layout invariance, no mutation, torch.compile vs eager, determinism), `inputs.py` (JSON input specs), `replay.py`, `conformance.py`, `z3tools.py`, `spaces.py` (each kernel's documented input domain) |
| `benchmark/` | `instances.json` (9 kernel@commit instances) and, per kernel, the **agent-visible** `task.md`, `entry.py` and `reference.py` |
| `baselines/` | `random_testing.py` (the random-testing baseline) |
| `harness_only/` | ground truth the agent never sees: `labels.json` (buggy/fixed, bug class, fix lines), issue and fix text, minimal repros, targeted specs, benchmark-construction records |
| `scripts/` | tree export and verification, GPU drivers (`gpu/`), harness tools (`harness/`), demo helpers |
| `tests/` | math checks of the four references (`pytest tests`) |
| `results/` | runs, traces, calibration, random testing, GPU logs |

The agent only ever gets a sandbox containing the exported tree, `task/` and the oracle helpers. Inside it there is no network, no file access outside the sandbox and no process spawning; `agent/sandbox.py` documents the guard.

## 8. Troubleshooting
- **`CERTIFICATE_VERIFY_FAILED` on macOS.** The python.org builds don't use the system keychain. The code already uses `truststore` for its own requests. For anything else, run `/Applications/Python 3.12/Install Certificates.command`.
- **`ModuleNotFoundError: triton` on macOS.** Run `scripts/install_triton_macos.sh`; PyPI has no macOS wheel.
- **bf16 results look wrong on CPU.** The Triton CPU interpreter computes bf16×bf16 arithmetic incorrectly. That's why bf16 is out of scope in the CPU-run kernels' `task.md` files; use a GPU for bf16.
- **`torch.compile` property reports `error` on CPU.** Dynamo can't trace the Triton interpreter. That property is GPU-only, and an error never counts as a violation.
- **Messages saying Liger backend imports failed (`No module named 'cuda'`).** Harmless. Liger falls back to its Triton backend when the CuTe/cuTile packages are absent.
