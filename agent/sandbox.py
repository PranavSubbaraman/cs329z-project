"""Per-run sandboxes for the agent under test.

Layout (everything the agent can see):
    <sandbox>/liger/     exported Liger-Kernel tree at ONE commit (read-only; mtimes reset to a constant)
    <sandbox>/task/      task.md, entry.py, reference.py, kernel_name.txt (read-only)
    <sandbox>/oracle/    the oracle helpers, importable as `oracle` (read-only)
    <sandbox>/scratch/   the only writable directory; cwd of every execution; per-run Triton/inductor caches
    <sandbox>/_boot/     sitecustomize.py: the in-process guard (not listed to the agent)

Every execution is a fresh subprocess with a timeout, cwd = scratch, and PYTHONPATH = _boot:liger/src:<sandbox>.
The guard is an audit hook installed by sitecustomize and inherited by child Python processes. It denies:
  * network: socket connect/bind/getaddrinfo/sendto (on Linux the process also runs in a fresh network namespace
    when `unshare -rn` is available);
  * file reads outside the sandbox and the Python/CUDA runtime, and any write outside scratch;
  * spawning anything except the Python interpreter itself and the compiler tools Triton needs (cc/gcc/ld/ptxas/ldconfig).
It is a guard against accidental access, not a security boundary against a determined attacker (ctypes can bypass
audit hooks); every executed snippet is logged for audit.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_MTIME = 946684800  # 2000-01-01T00:00:00Z: removes the commit-date side channel of git-archive mtimes

GUARD = r'''
import os, sys
_ROOT = os.environ.get("KV_SANDBOX_ROOT")
if _ROOT:
    _ROOT = os.path.realpath(_ROOT)
    _SCRATCH = os.path.join(_ROOT, "scratch")
    _BOOT = os.path.join(_ROOT, "_boot")
    _RUNTIME = [os.path.realpath(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}]
    _SYS = ["/dev", "/proc", "/sys", "/etc", "/usr", "/lib", "/lib64", "/bin", "/sbin", "/opt", "/run",
            "/System", "/Library", "/private/var/db", "/private/etc", "/var/run"]
    _EXTRA = [p for p in os.environ.get("KV_SANDBOX_EXTRA_READ", "").split(os.pathsep) if p]
    _READ = [_ROOT] + _RUNTIME + _SYS + [os.path.realpath(p) for p in _EXTRA]
    _WRITE = [_SCRATCH, "/dev/null", "/dev/shm", "/dev/nvidia", "/proc/self"]
    _EXEC_OK = ("cc", "gcc", "g++", "c++", "ld", "ld.bfd", "ld.gold", "ptxas", "nvlink", "fatbinary", "cuobjdump",
                "ldconfig", "ldconfig.real", "nvdisasm", "as", "x86_64-linux-gnu-gcc", "x86_64-linux-gnu-ld", "collect2", "cc1")
    def _under(p, roots):
        return any(p == r or p.startswith(r.rstrip("/") + "/") for r in roots)
    def _deny(msg):
        raise PermissionError("[sandbox] " + msg)
    def _hook(event, args):
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            if not isinstance(path, (str, bytes)) or path in ("", b""):
                return
            p = os.path.realpath(os.fsdecode(path))
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
            if writing and not _under(p, _WRITE):
                _deny("write outside scratch/: " + p)
            if not _under(p, _READ):
                _deny("read outside the sandbox: " + p)
        elif event in ("os.listdir", "os.scandir"):
            a0 = args[0] if args and args[0] is not None else "."
            if isinstance(a0, int):  # listing an already-open directory fd
                return
            p = os.path.realpath(os.fsdecode(a0))
            if not _under(p, _READ) or _under(p, [_BOOT]):
                _deny("listing outside the sandbox: " + p)
        elif event.startswith("socket.") and event in ("socket.connect", "socket.bind", "socket.getaddrinfo", "socket.sendto", "socket.gethostbyname"):
            _deny("network access is disabled")
        elif event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn"):
            exe = args[0] if args else ""
            exe = os.fsdecode(exe) if isinstance(exe, (str, bytes)) else ""
            base = os.path.basename(exe)
            real = os.path.realpath(exe) if os.path.isabs(exe) else exe
            if not (real == os.path.realpath(sys.executable) or base.startswith("python") or base in _EXEC_OK
                    or (os.path.isabs(exe) and _under(real, _RUNTIME))):
                _deny("process spawning is disabled: " + exe)
        elif event in ("os.system",):
            _deny("os.system is disabled")
    sys.addaudithook(_hook)
'''


def _copy_tree(src: Path, dst: Path):
    def ignore(d, names):
        return [n for n in names if n in ("__pycache__", ".pytest_cache") or n.endswith(".pyc")]
    shutil.copytree(src, dst, symlinks=True, ignore=ignore)


def _freeze(path: Path, writable: bool = False):
    for p in sorted(path.rglob("*"), key=lambda x: -len(x.parts)) + [path]:
        try:
            os.utime(p, (FIXED_MTIME, FIXED_MTIME), follow_symlinks=False)
        except (NotImplementedError, OSError):
            pass
        if not writable and not p.is_symlink():
            mode = p.stat().st_mode
            os.chmod(p, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


class Sandbox:
    def __init__(self, instance: dict, base: Path | None = None, backend: str | None = None, keep: bool = True):
        self.instance = instance
        self.kernel = instance["kernel"]
        self.commit = instance["commit"]
        self.run_id = uuid.uuid4().hex[:12]
        self.dir = (base or ROOT / "work" / "sandboxes") / self.run_id
        self.backend = backend or ("cuda" if _cuda_available() else "interpreter")
        self.exec_seconds = 0.0
        self.exec_log: list[dict] = []
        self._build()

    def _build(self):
        tree = ROOT / "work" / "trees" / self.commit
        if not (tree / "src" / "liger_kernel").is_dir():
            raise FileNotFoundError(f"exported tree missing: {tree} (run scripts/export_trees.py)")
        self.dir.mkdir(parents=True)
        _copy_tree(tree, self.dir / "liger")
        tdir = self.dir / "task"
        tdir.mkdir()
        for f in ("task.md", "entry.py", "reference.py"):
            shutil.copy2(ROOT / "benchmark" / "kernels" / self.kernel / f, tdir / f)
        (tdir / "kernel_name.txt").write_text(self.kernel + "\n")
        _copy_tree(ROOT / "oracle", self.dir / "oracle")
        (self.dir / "_boot").mkdir()
        (self.dir / "_boot" / "sitecustomize.py").write_text(GUARD)
        (self.dir / "scratch").mkdir()
        for sub in ("liger", "task", "oracle", "_boot"):
            _freeze(self.dir / sub)
        _freeze(self.dir / "scratch", writable=True)
        os.utime(self.dir, (FIXED_MTIME, FIXED_MTIME))

    # ---- path helpers for the read tools --------------------------------------------------------------------------
    def resolve(self, rel: str) -> Path:
        rel = (rel or ".").lstrip("/")
        p = (self.dir / rel).resolve()
        if not (p == self.dir.resolve() or str(p).startswith(str(self.dir.resolve()) + os.sep)):
            raise PermissionError(f"path outside the sandbox: {rel}")
        if p.parts[len(self.dir.resolve().parts):][:1] == ("_boot",):
            raise PermissionError("not part of the task")
        return p

    # ---- execution -----------------------------------------------------------------------------------------------
    def env(self) -> dict:
        sb = str(self.dir)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENROUTER", "KV_", "LIGER_"))}
        env.update({
            "PYTHONPATH": os.pathsep.join([f"{sb}/_boot", f"{sb}/liger/src", sb]),
            "KV_SANDBOX_ROOT": sb, "KV_TASK_DIR": f"{sb}/task", "KV_KERNEL": self.kernel,
            "PYTHONDONTWRITEBYTECODE": "1", "HOME": f"{sb}/scratch", "TMPDIR": f"{sb}/scratch",
            "TRITON_CACHE_DIR": f"{sb}/scratch/.triton", "TORCHINDUCTOR_CACHE_DIR": f"{sb}/scratch/.inductor",
            "TRITON_INTERPRET": "1" if self.backend == "interpreter" else "0",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"), "MPLCONFIGDIR": f"{sb}/scratch/.mpl",
        })
        env.pop("PYTHONSTARTUP", None)
        return env

    def run_python(self, code: str, timeout: float = 180.0, label: str = "snippet") -> dict:
        n = len(self.exec_log)
        path = self.dir / "scratch" / f"_{label}_{n:03d}.py"
        path.write_text(code)
        cmd = [sys.executable, str(path)]
        if sys.platform.startswith("linux") and _unshare_ok():
            cmd = ["unshare", "-rn"] + cmd
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=self.dir / "scratch", env=self.env(), capture_output=True, text=True,
                               timeout=timeout)
            rc, out, err, to = p.returncode, p.stdout, p.stderr, False
        except subprocess.TimeoutExpired as e:
            rc, out, err, to = -9, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), \
                (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or ""), True
        dt = time.time() - t0
        self.exec_seconds += dt
        rec = {"label": label, "file": path.name, "seconds": round(dt, 3), "returncode": rc, "timed_out": to}
        self.exec_log.append(rec)
        return {**rec, "stdout": out, "stderr": _clean_stderr(err)}

    def cleanup(self):
        for p in self.dir.rglob("*"):
            try:
                os.chmod(p, stat.S_IRWXU, follow_symlinks=False) if not p.is_symlink() else None
            except OSError:
                pass
        os.chmod(self.dir, stat.S_IRWXU)
        shutil.rmtree(self.dir, ignore_errors=True)


def _clean_stderr(err: str) -> str:
    keep = [l for l in err.splitlines() if not (l.startswith("W1") or "Redirects are currently not supported" in l
                                                 or "could not be imported: ModuleNotFoundError: No module named 'cuda'" in l)]
    return "\n".join(keep)


_UNSHARE = None


def _unshare_ok() -> bool:
    global _UNSHARE
    if _UNSHARE is None:
        try:
            _UNSHARE = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
        except Exception:
            _UNSHARE = False
    return _UNSHARE


def _cuda_available() -> bool:
    if os.environ.get("TRITON_INTERPRET", "0") in ("1", "true"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
