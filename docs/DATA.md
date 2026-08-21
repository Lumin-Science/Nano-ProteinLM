# Data contract

The current source payload is the verified Step-9 transfer at
`~/workspace/esmc-open-step9-clusters-v1/clusters` on tmoss:

| Source | Unique sequences | 70%-identity clusters | Representative FASTA |
|---|---:|---:|---:|
| UniRef90 | 165,884,293 | 92,230,941 | 38.6 GB |
| MGnify | 611,788,129 | 348,135,082 | 93.3 GB |
| OMG/IMG (JGI role) | 963,673,186 | 324,923,979 | 109.7 GB |

Only the representative FASTAs and membership TSVs were transferred. The TSVs
name cluster members but do not contain their sequences. Consequently, this
repository samples one transferred representative uniformly per cluster. It
does not claim to reproduce ESMC's cluster-then-member sampling.

`scripts/prepare_data.py` scans SHA-sorted representatives, verifies sequence
digests, applies a SHA-modulus validation split, excludes exact hashes present
in the 108,215-sequence P-CORE index and 20,795-chain P@L manifest, and writes
uint8 token mmaps with immutable receipts. SHA ordering makes a bounded prefix a
pseudorandom sample with respect to sequence content.

Remaining release blockers:

1. Homology-level decontamination against every benchmark split, ideally with
   MMseqs2 components at the benchmark's strictest identity/coverage policy.
2. Source-license and redistribution review before publishing derived shards.
3. Full-corpus length, ambiguity, taxonomy, and source-duplication audits.
4. A scalable member-sequence store if exact cluster-then-member sampling is a
   scientific requirement.
