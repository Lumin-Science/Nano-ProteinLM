# Data contract

## Public release

The supported training artifact is public at
[`LuminScience/LuminBench-Nano-ESMC`](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC).
Production defaults pin commit
[`bd38448d`](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/tree/bd38448d50d8f426d7b9bd4410b53159ea001259)
rather than a moving branch. Its verified manifest contains 665,970,495
globally exact-unique training representatives:

| Released source arm | Training records |
|---|---:|
| UniRef90 | 74,175,974 |
| MGnify | 328,949,335 |
| OMG/IMG | 262,845,186 |

Every source also carries one 4,096-record validation shard. The public Hub
artifact occupies 109,661,312,410 bytes, but a training run downloads only the
smallest checksum-bound shard prefix that covers its requested sequence
exposures.

The source databases and their original publications are UniRef90
([Suzek et al., 2015](https://doi.org/10.1093/bioinformatics/btu739)), MGnify
([Richardson et al., 2023](https://doi.org/10.1093/nar/gkac1080)), and the
OMG distribution's IMG/JGI-labelled arm
([Cornman et al., 2024](https://doi.org/10.1101/2024.08.14.607850)). This is an
open reconstruction of the data roles described by ESMC, not a claim that the
OMG/IMG arm reproduces Biohub's original private JGI corpus.

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
2.55-GiB Stage-1 subset and its experiment receipts were removed from the
supported tree because they predate the Q9 evaluation-union screen; their
provenance remains recoverable from repository history. They are not supported
or publishable training corpora.

Preparation verifies sequence hashes, excludes exact evaluation matches, runs
the safe evaluation-query all-splits MMseqs2 screen, applies the
32--16,384-residue filter,
constructs globally disjoint validation sets, and independently re-reads every
released row. The homology contract is 30% identity with both 80% query and 80%
target coverage. It protects contact P@L, P-CORE v0.2, and every P-CORE
v0.5-alpha-q9 sequence, including blocked future evaluation candidates.
An exact digest retained as a representative by multiple source-specific
clusters is assigned to one deterministic source owner before release, so the
36:11:54 sampler cannot overweight cross-source duplicates.

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

## Source terms

Lumin Science's selection, arrangement, decontamination ledger, packing, and
release metadata are distributed under CC BY-SA 4.0. Upstream records retain
their own terms: UniRef90 is CC BY 4.0; the direct OMG distribution declares CC
BY-SA 4.0; MGnify remains under the EMBL-EBI Terms of Use plus applicable
original-owner rights. Tokenization and mmap materialization do not relicense
the underlying sequences. The complete notices and path-specific attribution
ship beside the Hub manifest.
