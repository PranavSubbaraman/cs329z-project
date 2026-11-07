"""Gradient oracle and property checks for Triton training kernels.

    grad_compare(kernel_fn, ref_fn, inputs)        kernel (native dtype) vs reference (float64): outputs + input grads
    check_properties(kernel_fn, ref_fn, spec, kernel)   zero upstream / layout / no mutation / compile / determinism
    replay(assignment)                              concrete counterexample -> tensors -> does it reproduce a violation?
    conformance_check(model, kernel_fn, ref_fn, kernel)   does a hand-written model behave like the kernel?
    build_inputs(spec), example_spec(kernel), sample(kernel, rng, seed)    input specs (see oracle.inputs)
    load_task(), kernel_name()                      the task's entry point + reference (sandbox: ./task)
The tolerance rule is documented in oracle.tolerances.
"""

from .compare import grad_compare
from .conformance import conformance_check
from .inputs import build_inputs, clone_like, place
from .kernels import kernel_name, load_task, task_dir
from .properties import PROPERTIES, check_properties
from .replay import replay
from .spaces import ARGS, diff_args, example_spec, sample
from .tolerances import allowed_error, describe as tolerance_rule

__all__ = ["grad_compare", "check_properties", "replay", "conformance_check", "build_inputs", "clone_like", "place",
           "load_task", "kernel_name", "task_dir", "PROPERTIES", "ARGS", "diff_args", "example_spec", "sample",
           "allowed_error", "tolerance_rule"]
