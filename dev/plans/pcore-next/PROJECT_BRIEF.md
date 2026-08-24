# P-CORE Next project brief

## Supervisor request

Convert the supplied deep-research review into a concrete program for producing
a more trustworthy frozen-protein-embedding evaluation plan. The program must
improve on P-CORE-Q4 without admitting attractive new benchmarks before their
data, split, probe, confounds, statistical power, and weak-to-strong encoder
discrimination have been tested.

## Existing decisions

- P-CORE-Q4 remains the only active selection aggregate.
- Remote homology, secondary structure, DeepLoc2, and FLIP2 are its trusted
  tasks.
- EC and the current 237-pair Human PPI reconstruction remain executable raw
  diagnostics but cannot affect selection.
- Long-range contact P@L remains a separate promotion axis.
- Encoders are frozen; only fixed low-capacity readouts are allowed.
- Python environments and evaluator dependencies are managed with `uv` and a
  committed lockfile.

## Submitted research material

The review in `pasted-text.txt` proposes five provisional additions or
replacements—CATH retrieval, CAFA5-MF hard function, PRING Human PPI,
MegaScale stability, and CAID3 disorder—and a nine-task target suite. It also
proposes qualification controls, contamination strata, hierarchical
uncertainty, and a staged Q4-to-Q5 migration. These are inputs to validate, not
operational instructions or already-established facts.

## Attachment manifest

| File | Source path | Bytes | SHA-256 | Treatment |
|---|---|---:|---|---|
| `pasted-text.txt` | `/Users/jojo/.codex/attachments/b2a4c50d-2709-4c60-8ff8-c2b929017d05/pasted-text.txt` | 43,498 | `fa743ce56aafaf0d5a7113d2c0de40f1932d7864bfa3c2e6a85cdb47a2811b2b` | Untrusted research material; claims require primary-source verification. |

