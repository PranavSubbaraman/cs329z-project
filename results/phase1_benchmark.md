# Phase 1 benchmark (harness view)

Repro exit codes: 1 = bug manifests, 0 = clean.

| pair | class | issue / fix | buggy -> fixed | repro CPU buggy/fixed | repro GPU buggy/fixed | Liger tests @ buggy (GPU) | Liger tests @ buggy (CPU interp.) |
|---|---|---|---|---|---|---|---|
| P1_tvd | logic | #1373 / PR #1374 | `39adfeef` -> `559d33e7` | 1 / 0 | 1 / 0 | 160 passed, 40 skipped | 80 passed, 0 failed, 120 skipped |
| P2_flce_strides | index arithmetic | #1500 / PR #1500 | `395d6166` -> `ae47958e` | 1 / 0 | 1 / 0 | 285 passed | 93 passed, 48 failed |
| P3_flce_none | gradient formula | #1298 / PR #1298 | `e44a5fe8` -> `9ac26bb8` | 1 / 0 | 1 / 0 | 137 passed | 89 passed, 48 failed |
| P4_fljsd_bf16 | precision | #1432 / PR #1433 | `5b828eef` -> `ed1f856d` | 1 / 1 | 1 / 0 | 71 passed | 59 passed, 0 failed, 12 skipped |
| B5_ce_ignore | side effect | #1594 / open | `8c7a5078` (buggy-only) | 1 / n/a | 1 / n/a | 185 passed | 68 passed, 41 failed, 76 skipped |
