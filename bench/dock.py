"""dock.py -- run commands inside the pinned MP-SPDZ container used by the benchmark."""
import os, subprocess

IMAGE = "mldsa-bench:0.4.3"
NAME = "mldsa-bench"
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
MPSPDZ = "/opt/mp-spdz"


def _docker(*args, **kw):
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kw)


def ensure_container():
    """Start the long-lived container (repo bench/ mounted at /bench) if it is not running."""
    st = _docker("inspect", "-f", "{{.State.Running}}", NAME)
    if st.returncode == 0 and st.stdout.strip() == "true":
        return
    if st.returncode == 0:
        _docker("rm", "-f", NAME)
    r = _docker("run", "-d", "--name", NAME, "--cap-add", "NET_ADMIN", "--shm-size", "2g",
                "-v", f"{BENCH_DIR}:/bench", IMAGE, "sleep", "infinity")
    if r.returncode != 0:
        raise RuntimeError(f"docker run failed: {r.stderr}")


def sh(cmd, timeout=7200, check=True):
    """Run a bash command in the MP-SPDZ directory of the container."""
    r = subprocess.run(["docker", "exec", "-w", MPSPDZ, NAME, "bash", "-c", cmd],
                       capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n--- stdout\n{r.stdout[-4000:]}"
                           f"\n--- stderr\n{r.stderr[-4000:]}")
    return r


def put_text(path, text):
    """Write a text file inside the container (path relative to the MP-SPDZ directory)."""
    r = subprocess.run(["docker", "exec", "-i", "-w", MPSPDZ, NAME, "bash", "-c",
                        f"mkdir -p \"$(dirname '{path}')\" && cat > '{path}'"],
                       input=text, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"writing {path} failed: {r.stderr}")
