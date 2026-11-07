#!/usr/bin/env bash
# Phase 1 validation on a CUDA GPU (real Triton kernels: TRITON_INTERPRET=0). Run inside a GPU allocation:
#   srun --jobid=<job> --overlap scripts/gpu/phase1_gpu.sh [step ...]     steps: trees repros refs libtests net (default all)
# Outputs: results/gpu/ (env, repro JSONL, reference-agreement summaries, Liger test logs).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-gpu/bin/python"
OUT="$ROOT/results/gpu"; mkdir -p "$OUT/logs"
export TRITON_INTERPRET=0 PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$(dirname "$ROOT")/.cache/triton}"
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(trees repros refs libtests net)
has() { local s; for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
sha() { ls -d "$ROOT"/work/trees/"$1"* | head -1; }

if has trees; then
  "$PY" scripts/harness/verify_trees.py > "$OUT/verify_trees.txt" 2>&1; echo "verify_trees exit=$?" >> "$OUT/verify_trees.txt"
  cat "$OUT/verify_trees.txt"
fi

if has repros; then
  : > "$OUT/phase1_repros.jsonl"
  for spec in P1_tvd:39adfeef P1_tvd:559d33e7 P2_flce_strides:395d6166 P2_flce_strides:ae47958e \
              P3_flce_none:e44a5fe8 P3_flce_none:9ac26bb8 P4_fljsd_bf16:5b828eef P4_fljsd_bf16:ed1f856d B5_ce_ignore:8c7a5078; do
    pair="${spec%%:*}"; c="${spec##*:}"; t0=$(date +%s)
    tmp="$(mktemp)"
    scripts/harness/in_tree.sh "$c" "$PY" "harness_only/repros/$pair.py" > "$tmp" 2>>"$OUT/logs/repros_stderr.log"
    rc=$?   # the repro's own exit code: 1 = bug manifests, 0 = clean, 2 = error
    line="$(grep '^{' "$tmp" | tail -1)"; rm -f "$tmp"
    echo "{\"pair\": \"$pair\", \"commit\": \"$c\", \"exit_code\": $rc, \"seconds\": $(( $(date +%s) - t0 )), \"stdout\": ${line:-null}}" >> "$OUT/phase1_repros.jsonl"
    echo "repro $pair @ $c exit=$rc ($(( $(date +%s) - t0 ))s)"
  done
fi

if has refs; then
  for spec in tvd:39adfeef tvd:559d33e7 flce:ae47958e:all flce:9ac26bb8:contiguous flce:395d6166:all flce:e44a5fe8:contiguous \
              fljsd:5b828eef fljsd:ed1f856d ce:8c7a5078; do
    IFS=: read -r k c lay <<<"$spec"
    args=(); [ -n "${lay:-}" ] && args=(--layouts "$lay" --out "$OUT/flce_ref_vs_kernel_${c}_${lay}.json")
    [ "$k" = fljsd ] && args=(--out "$OUT/fljsd_ref_vs_kernel_${c}.json")
    t0=$(date +%s)
    scripts/harness/in_tree.sh "$c" "$PY" "scripts/harness/ref_vs_kernel_$k.py" "${args[@]}" > "$OUT/logs/ref_${k}_${c}${lay:+_$lay}.log" 2>&1
    rc=$?
    case "$k" in tvd|ce) cp "harness_only/phase1/${k}_ref_vs_kernel.json" "$OUT/${k}_ref_vs_kernel_${c}.json" 2>/dev/null ;; esac
    echo "ref_vs_kernel $k @ $c ${lay:-} exit=$rc ($(( $(date +%s) - t0 ))s)"
  done
fi

if has libtests; then
  for spec in "tvd:39adfeef:test/transformers/test_tvd.py" "tvd:559d33e7:test/transformers/test_tvd.py" \
              "flce:395d6166:test/transformers/test_fused_linear_cross_entropy.py test/ops/test_fused_linear_cross_entropy.py" \
              "flce:ae47958e:test/transformers/test_fused_linear_cross_entropy.py test/ops/test_fused_linear_cross_entropy.py" \
              "flce:e44a5fe8:test/transformers/test_fused_linear_cross_entropy.py" \
              "flce:9ac26bb8:test/transformers/test_fused_linear_cross_entropy.py" \
              "fljsd:5b828eef:test/transformers/test_fused_linear_jsd.py test/ops/test_fused_linear_jsd.py" \
              "fljsd:ed1f856d:test/transformers/test_fused_linear_jsd.py test/ops/test_fused_linear_jsd.py" \
              "ce:8c7a5078:test/transformers/test_cross_entropy.py"; do
    IFS=: read -r k c files <<<"$spec"
    tree="$(sha "$c")"; log="$OUT/logs/libtests_${k}_${c}.log"; t0=$(date +%s)
    existing=(); for f in $files; do [ -f "$tree/$f" ] && existing+=("$f"); done
    { echo "# commit $c tree $tree files ${existing[*]}"; echo "# started $(date -u +%FT%TZ)"; } > "$log"
    (cd "$tree" && PYTHONPATH=src "$PY" -m pytest "${existing[@]}" -q -p no:cacheprovider -o log_cli=false \
        --timeout 900 -rfEs --junitxml="$OUT/logs/libtests_${k}_${c}.xml") >> "$log" 2>&1
    rc=$?
    echo "# exit_code $rc wall_seconds $(( $(date +%s) - t0 ))" >> "$log"
    echo "libtests $k @ $c exit=$rc: $(grep -E '(passed|failed|error)' "$log" | grep -E ' in [0-9.]+s' | tail -1)"
  done
fi

if has net; then
  curl -s -o /dev/null -w "openrouter models endpoint from $(hostname): HTTP %{http_code}\n" --max-time 15 \
    https://openrouter.ai/api/v1/models | tee "$OUT/net_check.txt"
fi
