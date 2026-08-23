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

## Only supported prepared corpus

`stage1-300m-production-v1` contains 3,000,000 training and 4,096 validation
representatives from each source: 9,012,288 records and 2,345,995,606 residues
in total.

Preparation performs, in order:

1. header-to-sequence SHA-256 verification;
2. exact exclusion against every indexed P-CORE and P@L sequence;
3. MMseqs2 exclusion against all evaluation fit, validation, and test splits;
4. the 32--16,384-residue length filter;
5. deterministic SHA-modulus train/validation assignment; and
6. independent post-write intersection and file-hash verification.

The MMseqs2 contract uses at least 30% sequence identity, 80% query coverage,
and 80% target coverage. Its query set contains 116,841 unique evaluation
sequences, including the 16 P@L probe-fit, 4 probe-validation, and 20,775 test
chains. It excludes 1,592,566 candidate training representatives by homology.

The final verifier reports zero exact/homology intersections in every source,
zero validation contamination, and zero train-validation overlap. Compact
receipts are tracked in
[`results/stage1-300m-production-v1/`](../results/stage1-300m-production-v1/).

## Fail-closed behavior

`scripts/prepare_data.py` requires the homology exclusion digest set and receipt;
there is no exact-only mode. `nano_protein.train` independently requires the
all-splits homology contract and a matching `CORPUS_VERIFICATION.json` for every
training configuration.

Raw sources and prepared mmap shards remain in controlled storage. Publishing
derived data requires a source-by-source provenance and license review.
