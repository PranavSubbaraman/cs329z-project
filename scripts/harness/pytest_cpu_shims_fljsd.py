"""HARNESS-ONLY pytest plugin for kernel fljsd. Load with  -p pytest_cpu_shims -p pytest_cpu_shims_fljsd
(scripts/harness on PYTHONPATH). Used only for the extra "context" run recorded in harness_only/phase1/fljsd_notes.md,
never for the main library-test counts.

Shim (the only one): remove the CUDA-only `skipif` marker from
test/transformers/test_fused_linear_jsd.py::test_logits_projection_is_not_rounded_to_input_dtype, the regression test
that exists at the fixed commit ed1f856d. Why: the float32-output projection that test checks only takes effect on CUDA
sm_80+, so pytest skips it on CPU. Un-skipping it lets it RUN on CPU to document that, off CUDA, the fixed commit still
computes the old projection (the test is expected to fail on CPU at both commits). It changes no test logic and no
library code; the test's own internal pytest.skip (float16, beta=0.5) still applies.
"""

TARGET = "test_logits_projection_is_not_rounded_to_input_dtype"


def pytest_report_header(config):
    # Not a shim: records which liger_kernel source tree the run imported (used for the cross-tree context run).
    import liger_kernel

    return f"liger_kernel imported from: {liger_kernel.__file__}"


def pytest_collection_modifyitems(config, items):
    for item in items:
        if getattr(item, "originalname", item.name) == TARGET:
            item.own_markers[:] = [m for m in item.own_markers if m.name != "skipif"]
