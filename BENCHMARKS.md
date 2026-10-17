# RNA-MoDISco optimization benchmark

The optimization port was benchmarked on 2026-10-16 against RNA-MoDISco
commit `80c73a8`. The optimized implementation uses upstream TF-MoDISco
commit `28154f1` as its reference.

The large 1,868 x 12,288 RNA fixture described in the implementation plan was
not present in this workspace. The available fixed fixture contains 300 RNAs
of length 300. To exercise both metaclusters, the attribution signs of the last
150 RNAs were inverted. Both implementations used 150 maximum seqlets per
metacluster, a 20-nt sliding window, 5-nt flank, 30-nt final trim, 10-nt
initial flank, a 0.05 target FDR, and two Leiden runs. The Numba caches were
warm, and `/usr/bin/time -v` measured the complete Python process, including
imports and result construction.

| Thread limit | Implementation | Wall time | Peak RSS | Pattern seqlet counts |
|---|---|---:|---:|---|
| 1 | RNA baseline `80c73a8` | 60.27 s | 462,104 KB | +: 43/41/24; -: 48/37/22 |
| 1 | Optimized | 35.02 s | 346,524 KB | +: 43/41/24; -: 48/37/22 |
| 4 | RNA baseline `80c73a8` | 60.75 s | 467,396 KB | +: 43/41/24; -: 48/37/22 |
| 4 | Optimized | 35.54 s | 354,212 KB | +: 43/41/24; -: 48/37/22 |

On this small fixture, the optimized implementation is 1.72x faster with one
thread and 1.71x faster with four threads. Peak RSS is 25.0% lower with one
thread and 24.2% lower with four threads. The small workload does not expose
the expected parallel scaling of the 20k-50k-seqlet production workloads.

In addition to the counts above, the regression test hashes every pattern's
ordered coordinates, sequence, contribution scores, hypothetical contribution
scores, and subcluster assignments. Those hashes are exactly equal to the
baseline results.
