"""HARNESS-ONLY pytest plugin (flce builder): print which liger_kernel source file is imported.

Used for the cross-tree "transplant" run (a fixed tree's test file against a buggy tree's src), where the tree's
pyproject `pythonpath = ["src", "."]` would otherwise silently put the test tree's own src first.
Load with: -p flce_where   (scripts/harness on PYTHONPATH)
"""


def pytest_sessionstart(session):
    import liger_kernel.ops.fused_linear_cross_entropy as m

    print("\nLIGER_KERNEL_IMPORTED_FROM", m.__file__)
