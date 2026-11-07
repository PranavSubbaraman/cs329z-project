"""HARNESS-ONLY: verify each exported tree in work/trees/ is byte-identical to `git archive <commit>`.

For every instance commit, compares the tree's files against `git ls-tree -r <commit>` using git blob hashes
(sha1 of "blob <len>\\0" + content). Reports missing, extra (e.g. __pycache__) and modified files. Exit 1 on any
mismatch. Extra files matter: anything present in only some trees (bytecode caches, logs) could reveal whether a
tree is the buggy or the fixed side.

    python scripts/harness/verify_trees.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLONE = ROOT / ".harness_cache" / "liger-kernel"
TREES = ROOT / "work" / "trees"


def git_manifest(commit: str) -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(CLONE), "ls-tree", "-r", "-z", commit], check=True,
                         capture_output=True).stdout.decode()
    m = {}
    for entry in filter(None, out.split("\0")):
        meta, path = entry.split("\t", 1)
        mode, typ, sha = meta.split()
        if typ == "blob":
            m[path] = sha
    return m


def blob_sha(p: Path) -> str:
    data = p.read_bytes() if not p.is_symlink() else str(p.readlink()).encode()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tree_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file() or p.is_symlink()}


def main() -> int:
    inst = json.loads((ROOT / "benchmark" / "instances.json").read_text())["instances"]
    bad = 0
    for it in inst:
        tree = TREES / it["commit"]
        want = git_manifest(it["commit"])
        have = tree_files(tree)
        missing = sorted(set(want) - set(have))
        extra = sorted(set(have) - set(want))
        modified = sorted(p for p in set(want) & set(have) if blob_sha(have[p]) != want[p])
        ok = not (missing or extra or modified)
        bad += not ok
        print(f"{it['instance_id']:<16} files={len(have):<5} {'OK' if ok else 'MISMATCH'}"
              + ("" if ok else f" missing={len(missing)} extra={len(extra)} modified={len(modified)} "
                 f"e.g. {(missing + extra + modified)[:3]}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
