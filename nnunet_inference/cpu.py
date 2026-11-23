"""CPU and memory contract of the packaged model: the one place for these numbers.

entrypoint.sh, the Dockerfile labels and smoke-test.sh repeat them, because
they are read before Python starts; tests/test_cpu_contract.py keeps them equal.
"""
import os

MAX_THREADS = 16        # logical CPUs the container uses at most
THREADS_PER_FOLD = 3    # threads of each fold worker and of each numeric library
MEMORY_LIMIT_GIB = 64   # memory the container needs; its launcher enforces it
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")


def contract():
    """The numbers as the deployment manifest records them."""
    return dict(max_threads=MAX_THREADS, threads_per_fold=THREADS_PER_FOLD,
                memory_limit_gib=MEMORY_LIMIT_GIB)


def cpus_are_assigned(available):
    """Whether the launcher gave this container its own CPUs (a cpuset).

    With MAX_THREADS CPUs or fewer, each worker is tied to three of them.
    With more, none were assigned: the threads are left to the kernel, which
    runs them on CPUs that are free. Tying them to the lowest ones would put
    every container of a host on the same CPUs.
    """
    return len(available) <= MAX_THREADS


def set_thread_environment():
    """Limit the numeric libraries in this process and in the ones it starts."""
    for name in THREAD_VARIABLES:
        os.environ[name] = str(THREADS_PER_FOLD)
    os.environ["nnUNet_def_n_proc"] = str(THREADS_PER_FOLD)
    os.environ["nnUNet_compile"] = "false"
