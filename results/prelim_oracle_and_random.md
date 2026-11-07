## Oracle validation (GPU)

| instance | pair/side | targeted spec flagged | violated checks | calibration violations / n | max error/allowed |
|---|---|---|---|---|---|
| tvd@39adfeef | P1_tvd buggy | yes | grad_compare | --  | -- |
| tvd@559d33e7 | P1_tvd fixed | no |  | 0/100  | 0.3402584224604043 |
| flce@395d6166 | P2_flce_strides buggy | yes | grad_compare, zero_upstream, layout_invariance | --  | -- |
| flce@ae47958e | P2_flce_strides fixed | no |  | 0/100  | 0.18313506500597243 |
| flce@e44a5fe8 | P3_flce_none buggy | yes | grad_compare | --  | -- |
| flce@9ac26bb8 | P3_flce_none fixed | no |  | 26/100 grad_compare:4, zero_upstream:9, layout_invariance:21, determinism:2 | 9782289.981520807 |
| fljsd@5b828eef | P4_fljsd_bf16 buggy | yes | grad_compare | --  | -- |
| fljsd@ed1f856d | P4_fljsd_bf16 fixed | no |  | 0/100  | 0.9604977483997011 |
| ce@8c7a5078 | B5_ce_ignore buggy | yes | no_mutation_without_grad | 60/100 no_mutation_without_grad:60 | 0.2970822664989176 |

## Preliminary random differential testing (GPU, 200 specs max, seed 0, stop at first violation)

| instance | pair/side | verdict | first violation (spec #) | specs run | GPU-s |
|---|---|---|---|---|---|
| tvd@39adfeef | P1_tvd buggy | bug | grad_compare (#38) | 39 | 18.0 |
| tvd@559d33e7 | P1_tvd fixed | no_bug | -- | 200 | 42.6 |
| flce@395d6166 | P2_flce_strides buggy | bug | zero_upstream (#10) | 11 | 13.7 |
| flce@ae47958e | P2_flce_strides fixed | bug | grad_compare (#195) | 196 | 245.2 |
| flce@e44a5fe8 | P3_flce_none buggy | bug | grad_compare (#0) | 1 | 11.7 |
| flce@9ac26bb8 | P3_flce_none fixed | bug | zero_upstream (#10) | 11 | 13.0 |
| fljsd@5b828eef | P4_fljsd_bf16 buggy | bug | grad_compare (#0) | 1 | 9.9 |
| fljsd@ed1f856d | P4_fljsd_bf16 fixed | no_bug | -- | 200 | 87.1 |
| ce@8c7a5078 | B5_ce_ignore buggy | bug | no_mutation_without_grad (#0) | 1 | 12.3 |
