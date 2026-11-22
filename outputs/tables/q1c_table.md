| Task | Method | Possible? | Reason |
|---|---|---|---|
| (i) count steady | Sum | Yes | rho uses coordinate 2 |
| (i) count steady | Mean | No | [0] and [0, 0] both give (0, 1, 0) but answers are 1 vs 2 |
| (i) count steady | Max | No | [0] and [0, 0] both give (0, 1, 0) but answers are 1 vs 2 |
| (ii) more than half slowing | Sum | Yes | rho uses s1 > s2 + s3 |
| (ii) more than half slowing | Mean | Yes | rho uses m1 > 1/2 |
| (ii) more than half slowing | Max | No | [-1, 1] and [-1, -1, 1] both give (1, 0, 1) but answers are False vs True |
| (iii) any slowing | Sum | Yes | rho uses s1 > 0 |
| (iii) any slowing | Mean | Yes | rho uses m1 > 0 |
| (iii) any slowing | Max | Yes | rho uses max1 = 1 |
