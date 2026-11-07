#!/usr/bin/env bash
# HARNESS-ONLY helper: run a command against one exported Liger-Kernel tree.
#   scripts/harness/in_tree.sh <commit-prefix> <command...>
# Sets PYTHONPATH=<tree>/src, LIGER_TREE=<tree>, TRITON_INTERPRET=1 (override with TRITON_INTERPRET=0 on a GPU),
# and PYTHONDONTWRITEBYTECODE=1 so the exported tree stays byte-identical to `git archive`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
prefix="$1"; shift
tree="$(ls -d "$ROOT"/work/trees/"$prefix"* 2>/dev/null | head -1)"
[ -n "$tree" ] || { echo "no exported tree for $prefix (run scripts/export_trees.py)" >&2; exit 2; }
export LIGER_TREE="$tree"
export PYTHONPATH="$tree/src${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_INTERPRET="${TRITON_INTERPRET:-1}"
export PYTHONDONTWRITEBYTECODE=1
exec "$@"
