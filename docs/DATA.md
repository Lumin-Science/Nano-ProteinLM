# Data contract

## Source reservoir

The controlled Step-9 transfer contains one representative per 70%-identity
cluster:

| Source | Unique sequences | 70%-identity clusters | Representative FASTA |
|---|---:|---:|---:|
| UniRef90 | 165,884,293 | 92,230,941 | 38.6 GB |
| MGnify | 611,788,129 | 348,135,082 | 93.3 GB |
| OMG/IMG | 963,673,186 | 324,923,979 | 109.7 GB |

Membership TSVs do not include member sequences. Training therefore samples
transferred representatives; it does not claim ESMC's cluster-then-member
sampling.

## Supported release

The supported corpus is `full-open-v2`: the complete transferred 70%-identity
representative reservoir after length filtering and decontamination. The older
2.55-GiB Stage-1 subset is retained only as a historical experiment record under
`dev/results/`; it is not a supported or publishable training corpus because it
predates the Q9 evaluation-union screen.

Preparation verifies sequence hashes, excludes exact evaluation matches, runs
the symmetric all-splits MMseqs2 screen, applies the 32--16,384-residue filter,
constructs globally disjoint validation sets, and independently re-reads every
released row. The homology contract is 30% identity with both 80% query and 80%
target coverage. It protects contact P@L, P-CORE v0.2, and every P-CORE
v0.5-alpha-q9 sequence, including blocked future evaluation candidates.

The distribution format is source/split-partitioned Parquet with `sequence`,
`sha256`, and `length` fields. Shards are ordered by sequence digest and sized by
an uncompressed 256 Mi-residue ceiling.

`scripts/download_data.py` resolves a requested Hugging Face revision to one
immutable commit, reads the verified release manifest, and downloads the
smallest whole-shard prefix for each source that covers a run's total planned
sequence exposures. All validation shards are always included. The command then
records protein, residue, and compressed-byte totals separately, rehashes each
row while materializing the existing mmap stores, and leaves training fully
local.

The v2 training gate additionally requires P@L, P-CORE v0.2, and P-CORE
v0.5-alpha-q9 in the homology-screen receipt, including Q9 tasks blocked from
headline scoring. A blocked benchmark is still a future evaluation candidate and
therefore not valid pretraining data.
