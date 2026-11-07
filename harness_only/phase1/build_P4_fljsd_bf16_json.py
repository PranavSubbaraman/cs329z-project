"""HARNESS-ONLY: assemble harness_only/phase1/P4_fljsd_bf16.json from the run artifacts in harness_only/phase1/logs/.
Re-run after a GPU run of run_repro_P4_fljsd_bf16.py (which overwrites logs/P4_fljsd_bf16_repro_runs.json).

    .venv/bin/python harness_only/phase1/build_P4_fljsd_bf16_json.py
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P1 = os.path.join(ROOT, "harness_only", "phase1")
LOGS = os.path.join(P1, "logs")
BUGGY = "5b828eef2bd2a723d001a7557b2e12da1817b505"
FIXED = "ed1f856d5b00d4ff75a62d66cdb78a4ad935d02f"
FILES = "test/transformers/test_fused_linear_jsd.py test/ops/test_fused_linear_jsd.py"
MAIN_CMD = ("cd work/trees/{sha} && TRITON_INTERPRET=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:{root}/scripts/harness "
            "{root}/.venv/bin/python -m pytest " + FILES + " -q -p no:cacheprovider -p pytest_cpu_shims "
            "-o log_cli=false -n 3 --timeout 600 -rs --durations=15")
CTX_CMD = ("cd work/trees/" + FIXED + " && TRITON_INTERPRET=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={pp} "
           "{root}/.venv/bin/python -m pytest test/transformers/test_fused_linear_jsd.py "
           "-k test_logits_projection_is_not_rounded_to_input_dtype -p no:cacheprovider -p pytest_cpu_shims "
           "-p pytest_cpu_shims_fljsd -o log_cli=false{extra} -n 3 --timeout 1200 -rA")


def counts(log_path):
    text = open(log_path).read()
    line = [ln for ln in text.splitlines() if re.search(r"\d+ (passed|failed|skipped)", ln) and " in " in ln][-1]
    get = lambda k: int(m.group(1)) if (m := re.search(rf"(\d+) {k}", line)) else 0  # noqa: E731
    secs = float(re.search(r" in ([0-9.]+)s", line).group(1))
    return {"passed": get("passed"), "failed": get("failed"), "skipped": get("skipped"), "errors": get("error"),
            "xfailed": get("xfailed"), "seconds": secs}, text


def main():
    runs = json.load(open(os.path.join(LOGS, "P4_fljsd_bf16_repro_runs.json")))
    lib = []
    for sha, note in ((BUGGY, "complete files, no subset. The 12 skips are all of test/ops/test_fused_linear_jsd.py "
                              "(its conftest autouse fixture skips every test without CUDA)."),
                      (FIXED, "complete files, no subset. Skips: the same 12 test/ops tests (CUDA required) + the 6 "
                              "parametrizations of the fix's new regression test "
                              "test_logits_projection_is_not_rounded_to_input_dtype (skipif: CUDA sm_80+, torch>=2.8).")):
        c, _ = counts(os.path.join(LOGS, f"fljsd_libtests_{sha[:8]}.log"))
        lib.append({"commit": sha, "files": FILES, "command": MAIN_CMD.format(sha=sha, root="<root>"), **c,
                    "subset_notes": note})
    for which, pp, extra, note in (
        ("fixed_src", "src:<root>/scripts/harness", "",
         "CONTEXT run, not part of the main counts: only the fix's regression test, un-skipped on CPU by the "
         "per-kernel shim scripts/harness/pytest_cpu_shims_fljsd.py (removes its CUDA-only skipif). It fails on the "
         "fixed commit on CPU because the float32-output projection is CUDA sm_80+ only. Its internal skip "
         "(float16, beta=0.5) still applies."),
        ("buggy_src", "<root>/scripts/harness", ' -o "pythonpath=work/trees/' + BUGGY + '/src ."',
         "CONTEXT run: the fixed commit's regression test file executed against the BUGGY commit's src (the tree's "
         "pyproject sets pytest pythonpath=[src, .], so it is overridden with -o; the imported liger_kernel path is "
         "printed in the log header and was checked). Same failures, same magnitudes as on the fixed src."),
    ):
        c, text = counts(os.path.join(LOGS, f"fljsd_regression_test_unskipped_{which}.log"))
        devs = sorted(set(re.findall(r"(grad_input|grad_weight) deviates ([0-9.]+%) from the float32-projection "
                                     r"reference, which means the projection rounded its logits to torch\.(\w+)", text)))
        lib.append({"commit": FIXED if which == "fixed_src" else f"{BUGGY} (src) with {FIXED} (test file)",
                    "files": "test/transformers/test_fused_linear_jsd.py::test_logits_projection_is_not_rounded_to_input_dtype",
                    "command": CTX_CMD.format(pp=pp, root="<root>", extra=extra), **c, "subset_notes": note,
                    "failure_messages": [f"{n} deviates {p} ({d})" for n, p, d in devs]})

    sim = json.load(open(os.path.join(P1, "P4_fljsd_bf16_sim.json")))["summary"]
    emu_path = os.path.join(P1, "P4_fljsd_bf16_patched_projection.json")
    emu = json.load(open(emu_path)) if os.path.exists(emu_path) else None
    b, f = runs["buggy"], runs["fixed"]
    def numeric(run):  # per-config metrics without the wall-time field
        return [{k: v for k, v in c.items() if k != "seconds_library"}
                for c in run["stdout_json"].get("metrics", {}).get("configs", [])]

    same = numeric(b) == numeric(f) and bool(numeric(b))
    backend = b["stdout_json"].get("backend")
    distinguishable = b["exit_code"] == 1 and f["exit_code"] == 0
    out = {
        "pair_id": "P4_fljsd_bf16",
        "buggy": {k: b[k] for k in ("stdout_json", "exit_code", "seconds")},
        "fixed": {k: f[k] for k in ("stdout_json", "exit_code", "seconds")},
        "reproduces_on_cpu": bool(distinguishable and backend == "interpreter-cpu"),
        "library_tests": lib,
        "notes": (
            f"Backend {backend}. The repro gives bug_manifest=true (exit {b['exit_code']}) on the buggy tree AND "
            f"bug_manifest=true (exit {f['exit_code']}) on the fixed tree; per-config metrics identical across trees: "
            f"{same}. This is the EXPECTED CPU outcome: the fix (_project_to_float32 -> torch.mm(..., "
            "out_dtype=float32)) only takes effect on CUDA with compute capability >= 8 (torch >= 2.8, same-dtype "
            "fp16/bf16 operands); off CUDA the fixed commit keeps the old (x @ w.t()).to(float32) expression, and "
            "aten::mm.dtype is not implemented on CPU at all. So the pair is NOT distinguishable on CPU; the real "
            "GPU run is pending (TRITON_INTERPRET=0 .venv/bin/python harness_only/phase1/run_repro_P4_fljsd_bf16.py, "
            "then rebuild this file). Threshold (2%) validated harness-side by a pure-torch SIMULATION over 5 seeds "
            f"(float32-output path <= {sim['max_criterion_float32_output']:.2%}, cast-after-matmul path >= "
            f"{sim['min_criterion_cast_after_matmul']:.2%}) and by an EMULATION that runs the fixed commit's library "
            "code on CPU with only _project_to_float32 monkeypatched to its CUDA branch"
            + (f" (repro exit {emu['repro_exit_code']}, max rel {emu['repro_stdout_json']['metrics']['max_grad_rel']:.2%})"
               if emu else "")
            + ". Liger's own suites pass at both commits on CPU (they cannot see this bug: their oracle rounds the "
              "same way); the fix's regression test is CUDA-gated and, un-skipped on CPU, fails identically at both."
        ),
        "harness_evidence": {
            "simulation": {"path": "harness_only/phase1/P4_fljsd_bf16_sim.json",
                           "script": "harness_only/phase1/P4_fljsd_bf16_sim.py",
                           "kind": sim["kind"], "threshold": sim["threshold"],
                           "criterion_per_seed": sim["criterion_per_seed"],
                           "margin_fixed_path": sim["margin_fixed_path"], "margin_buggy_path": sim["margin_buggy_path"],
                           "threshold_separates_paths_for_every_seed": sim["threshold_separates_paths_for_every_seed"]},
            "emulation_fixed_commit_float32_projection": (
                {"path": "harness_only/phase1/P4_fljsd_bf16_patched_projection.json",
                 "script": "harness_only/phase1/P4_fljsd_bf16_patched_projection.py", "kind": emu["kind"],
                 "repro_exit_code": emu["repro_exit_code"],
                 "max_grad_rel": emu["repro_stdout_json"]["metrics"]["max_grad_rel"],
                 "configs": [{k: c[k] for k in ("jsd_beta", "logit_std", "grad_input_rel", "grad_weight_rel")}
                             for c in emu["repro_stdout_json"]["metrics"]["configs"]]} if emu else None),
            "expected_on_ampere_plus_gpu": {"buggy": "exit 1 (max rel ~8-16%; simulation lower bound 8.8% at seed 0)",
                                            "fixed": "exit 0 (max rel ~0.5-0.7%)"},
        },
    }
    with open(os.path.join(P1, "P4_fljsd_bf16.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("pair_id", "reproduces_on_cpu")}),
          [(x["commit"][:8], x["passed"], x["failed"], x["skipped"]) for x in lib])


if __name__ == "__main__":
    main()
