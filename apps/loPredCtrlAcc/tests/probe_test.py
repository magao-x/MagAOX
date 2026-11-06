r"""\file probe_test.py
\brief Confirm that ordinary Eigen assertions and forbidden allocations still abort in isolated child processes.
"""
import resource
import signal
import subprocess
import sys


def no_core_dump():
    """Suppress large crash artifacts for deliberately failing assertion children."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


if __name__ == "__main__":
    executable = sys.argv[1]
    subprocess.run([executable], check=True)
    for option in ("--fail-assert", "--fail-allocation"):
        result = subprocess.run([executable, option], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, preexec_fn=no_core_dump)
        if result.returncode != -signal.SIGABRT:
            raise SystemExit(f"Assertion failure was not preserved: {option}: {result.returncode}")
    print("Ordinary assertions and allocation guards both retain SIGABRT")
