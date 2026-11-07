| instance | class | side | liger_tests | random | static | no_z3 | full |
|---|---|---|---|---|---|---|---|
| tvd@39adfeef | logic | buggy | pass | detection (39 specs, 18 GPU-s) | detection+loc ($0.12/run, n=1) |  | detection+loc ($0.19/run, n=1) |
| tvd@559d33e7 | logic | fixed | pass | correct_no_bug (200 specs, 43 GPU-s) | correct_no_bug ($0.13/run, n=1) |  | correct_no_bug ($0.28/run, n=1) |
| flce@395d6166 | index arithmetic | buggy | pass | detection (11 specs, 38 GPU-s) | detection ($0.46/run, n=1) |  | detection ($0.65/run, n=1) |
| flce@ae47958e | index arithmetic | fixed | pass | correct_no_bug (200 specs, 217 GPU-s) | correct_no_bug ($0.90/run, n=1) |  | correct_no_bug ($0.64/run, n=1) |
| flce@e44a5fe8 | gradient formula | buggy | pass | detection (1 specs, 13 GPU-s) | detection ($0.50/run, n=1) |  | detection ($0.38/run, n=1) |
| flce@9ac26bb8 | gradient formula | fixed | pass | flagged (replays) (11 specs, 19 GPU-s) | possible_new_bug ($0.34/run, n=1) |  | possible_new_bug ($1.03/run, n=1) |
| fljsd@5b828eef | precision | buggy | pass | detection (1 specs, 10 GPU-s) | miss ($1.00/run, n=1) |  | detection+loc ($0.77/run, n=1) |
| fljsd@ed1f856d | precision | fixed | pass | correct_no_bug (200 specs, 87 GPU-s) | correct_no_bug ($0.99/run, n=1) |  | correct_no_bug ($0.96/run, n=1) |
| ce@8c7a5078 | side effect | buggy | pass | detection (1 specs, 12 GPU-s) | miss ($0.43/run, n=1) |  | detection+loc ($0.75/run, n=1) |

| method | runs | detections | localized | false alarms | flagged real bug on fixed | proofs | false proofs | cost $ | mean GPU-s |
|---|---|---|---|---|---|---|---|---|---|
| static | 9 | 3/5 | 1/5 | 0/4 | 1 | 0 | 0 | 4.876 | 0.0 |
| full | 9 | 5/5 | 3/5 | 0/4 | 1 | 0 | 0 | 5.649 | 129.7 |
