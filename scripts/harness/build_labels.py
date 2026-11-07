"""HARNESS-ONLY: build harness_only/labels.json (ground truth) from the fix diffs.

Localization target = old-side (buggy-commit) line numbers touched by each src/ hunk of the fix:
  - hunk "-a,b" with b > 0: lines a .. a+b-1 (deleted or modified lines)
  - hunk "-a,0" (pure insertion after line a): anchor lines a and a+1
Test files are excluded. For the open issue (#1594) there is no fix diff; its target lines come from
the issue's own analysis and are marked source="issue".
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "harness_only"

PAIRS = [
    dict(pair_id="P1_tvd", kernel="tvd", issue=1373, fix_pr=1374, bug_class="logic",
         buggy="39adfeefab06c9a8b596199cb7f9e9d515986d8c", fixed="559d33e7b14d81476b295ef41a962937efa74517"),
    dict(pair_id="P2_flce_strides", kernel="flce", issue=1500, fix_pr=1500, bug_class="index arithmetic",
         buggy="395d6166e86f64f7c27f3c6e49cef89904add1aa", fixed="ae47958e39568874e3b21419130582a271f2a903"),
    dict(pair_id="P3_flce_none", kernel="flce", issue=1298, fix_pr=1298, bug_class="gradient formula",
         buggy="e44a5fe86d3ee00def43913329e4c4ff70c3b416", fixed="9ac26bb82226c9279f0fda784300f5e02ce444b2"),
    dict(pair_id="P4_fljsd_bf16", kernel="fljsd", issue=1432, fix_pr=1433, bug_class="precision",
         buggy="5b828eef2bd2a723d001a7557b2e12da1817b505", fixed="ed1f856d5b00d4ff75a62d66cdb78a4ad935d02f"),
    dict(pair_id="B5_ce_ignore", kernel="ce", issue=1594, fix_pr=None, bug_class="side effect",
         buggy="8c7a5078318da1295dc3ca445f9d85e9e9e153f5", fixed=None, status="open, buggy-only",
         fix_lines={"src/liger_kernel/ops/cross_entropy.py": [[104, 108]]}, fix_lines_source="issue"),
]


def old_side_lines(u0_diff: str):
    out, cur = {}, None
    for line in u0_diff.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
        if m and cur and cur.startswith("src/"):
            a, b = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            rng = [a, a + b - 1] if b > 0 else [a, a + 1]
            out.setdefault(cur, []).append(rng)
    return out


# Phase 1 outcome per pair (see harness_only/phase1/<pair_id>.json) and other known defects present at each
# commit, so a replayed claim on a "fixed" commit can be told apart from a possible new bug.
STATUS = {
    "P1_tvd": "kept: reproduces on CPU interpreter and GPU (RTX A4000)",
    "P2_flce_strides": "kept: reproduces on CPU interpreter and GPU (RTX A4000)",
    "P3_flce_none": "kept: reproduces on CPU interpreter and GPU (RTX A4000)",
    "P4_fljsd_bf16": "kept: reproduces on GPU (RTX A4000, sm_86); CPU interpreter cannot distinguish (fix is CUDA sm>=80 only)",
    "B5_ce_ignore": "kept (buggy-only, issue open): reproduces on CPU interpreter and GPU",
}
KNOWN_OTHER_BUGS = {
    "39adfeefab06c9a8b596199cb7f9e9d515986d8c": ["tvd all-rows-ignored mean/batchmean raises ZeroDivisionError (out of scope)"],
    "559d33e7b14d81476b295ef41a962937efa74517": ["tvd all-rows-ignored mean/batchmean raises ZeroDivisionError (out of scope)"],
    "e44a5fe86d3ee00def43913329e4c4ff70c3b416": ["P2 column-stride bug (#1500, unfixed until ae47958e)",
        "#1329 bf16/fp16 true-class gradient precision (out of scope: flce is float32-only)",
        "frozen input + trainable head gets no head gradient (open PR #1588; out of scope)",
        "retain_graph double backward rescales saved grads in place (out of scope)"],
    "9ac26bb82226c9279f0fda784300f5e02ce444b2": ["P2 column-stride bug (#1500, unfixed until ae47958e)",
        "#1329 bf16/fp16 true-class gradient precision (out of scope: flce is float32-only)",
        "frozen input + trainable head gets no head gradient (open PR #1588; out of scope)",
        "retain_graph double backward rescales saved grads in place (out of scope)"],
    "395d6166e86f64f7c27f3c6e49cef89904add1aa": ["frozen input + trainable head gets no head gradient (open PR #1588; out of scope)",
        "retain_graph double backward rescales saved grads in place (out of scope)"],
    "ae47958e39568874e3b21419130582a271f2a903": ["frozen input + trainable head gets no head gradient (open PR #1588; out of scope)",
        "retain_graph double backward rescales saved grads in place (out of scope)"],
    "5b828eef2bd2a723d001a7557b2e12da1817b505": ["element_mul_kernel column-stride bug (sibling of #1500, unfixed at HEAD; out of scope)",
        "AMP/autocast with fp32 head keeps logit rounding (out of scope)", "possible JSD underflow NaN (PR #1494 unmerged; unverified)"],
    "ed1f856d5b00d4ff75a62d66cdb78a4ad935d02f": ["element_mul_kernel column-stride bug (sibling of #1500, unfixed at HEAD; out of scope)",
        "AMP/autocast with fp32 head keeps logit rounding (out of scope)", "possible JSD underflow NaN (PR #1494 unmerged; unverified)"],
    "8c7a5078318da1295dc3ca445f9d85e9e9e153f5": ["by-design in-place gradient write into input when requires_grad=True (#272)",
        "input with stride(-1) != 1 gets the logits back as its gradient (no issue found; out of scope)"],
}

for p in PAIRS:
    p["phase1_status"] = STATUS[p["pair_id"]]
    p["known_other_bugs"] = {c[:8]: KNOWN_OTHER_BUGS.get(c, []) for c in (p["buggy"], p.get("fixed")) if c}
    if p.get("fixed"):
        p["fix_lines"] = old_side_lines((H / "fixes" / f"{p['issue']}.u0.diff").read_text())
        p["fix_lines_source"] = "fix diff (old side)"
        p.setdefault("status", "fixed")
(H / "labels.json").write_text(json.dumps({"pairs": PAIRS}, indent=2) + "\n")
for p in PAIRS:
    print(p["pair_id"], p["bug_class"], json.dumps(p["fix_lines"]))
