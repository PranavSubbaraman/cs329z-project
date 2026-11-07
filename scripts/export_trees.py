"""Export Liger-Kernel source trees (no .git) for benchmark commits into work/trees/<commit>/.

Uses `git archive` on the harness-side clone if present (.harness_cache/liger-kernel), otherwise
downloads GitHub's codeload tarball for the commit, which GitHub also produces with `git archive`.
Either way the exported tree has no .git directory, so no history, commit messages, or diffs.

    python scripts/export_trees.py               # all instances in benchmark/instances.json
    python scripts/export_trees.py --kernel tvd  # only one kernel's instances
"""

import argparse
import io
import json
import shutil
import ssl
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLONE = ROOT / ".harness_cache" / "liger-kernel"
TREES = ROOT / "work" / "trees"


def export(commit: str, dest: Path) -> str:
    if (dest / "src" / "liger_kernel").is_dir():
        return "cached"
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if (CLONE / ".git").is_dir():
        data = subprocess.run(["git", "-C", str(CLONE), "archive", "--format=tar", commit],
                              check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(tmp, filter="data")
        how = "git archive"
    else:
        url = f"https://codeload.github.com/linkedin/Liger-Kernel/tar.gz/{commit}"
        try:  # python.org macOS builds don't read the system keychain; truststore (an openai dependency) does
            import truststore
            ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
            data = r.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            tf.extractall(tmp, filter="data")
        (top,) = list(tmp.iterdir())  # codeload wraps everything in Liger-Kernel-<sha>/
        for p in top.iterdir():
            p.rename(tmp / p.name)
        top.rmdir()
        how = "codeload tarball"
    assert not (tmp / ".git").exists()
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return how


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=None)
    args = ap.parse_args()
    inst = json.loads((ROOT / "benchmark" / "instances.json").read_text())["instances"]
    for it in inst:
        if args.kernel and it["kernel"] != args.kernel:
            continue
        dest = TREES / it["commit"]
        print(f"{it['instance_id']:<16} {export(it['commit'], dest):<16} {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
