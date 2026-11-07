#!/usr/bin/env bash
# HARNESS-ONLY. Fetches issue/PR text and fix diffs for the benchmark into harness_only/.
# Nothing written here is ever copied into an agent sandbox.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO="linkedin/Liger-Kernel"
CLONE="$ROOT/.harness_cache/liger-kernel"
OUT="$ROOT/harness_only"
mkdir -p "$OUT/issues" "$OUT/fixes"
[ -d "$CLONE/.git" ] || git clone --quiet "https://github.com/$REPO.git" "$CLONE"

# issue/PR number -> fix commit (empty = no merged fix)
declare -a ITEMS=(
  "1373:559d33e7b14d81476b295ef41a962937efa74517:1374"
  "1500:ae47958e39568874e3b21419130582a271f2a903:1500"
  "1298:9ac26bb82226c9279f0fda784300f5e02ce444b2:1298"
  "1432:ed1f856d5b00d4ff75a62d66cdb78a4ad935d02f:1433"
  "1594::"
)
for item in "${ITEMS[@]}"; do
  IFS=: read -r num fix pr <<<"$item"
  gh api "repos/$REPO/issues/$num" --jq '"# #\(.number): \(.title)\nstate: \(.state)\nurl: \(.html_url)\ncreated: \(.created_at)\n\n\(.body // "")"' > "$OUT/issues/$num.md"
  gh api "repos/$REPO/issues/$num/comments?per_page=100" --jq '.[] | "\n---\ncomment by \(.user.login) at \(.created_at):\n\(.body)"' >> "$OUT/issues/$num.md"
  if [ -n "$pr" ] && [ "$pr" != "$num" ]; then
    gh api "repos/$REPO/pulls/$pr" --jq '"# PR #\(.number): \(.title)\nmerged_at: \(.merged_at)\nmerge_commit: \(.merge_commit_sha)\n\n\(.body // "")"' > "$OUT/issues/pr_$pr.md"
  fi
  if [ -n "$fix" ]; then
    git -C "$CLONE" show --format='commit %H%nparent %P%n%n%s%n%n%b' "$fix" > "$OUT/fixes/$num.diff"
    git -C "$CLONE" show --format= --unified=0 "$fix" > "$OUT/fixes/$num.u0.diff"
  fi
done
echo "wrote:"; ls -1 "$OUT/issues" "$OUT/fixes"
