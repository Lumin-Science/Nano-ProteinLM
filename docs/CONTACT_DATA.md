# Frozen contact evaluation data

NanoProtein's P@L setup bundle contains the exact normalized chain payloads and
frozen evaluator used by the [completed 100k-step comparison](../.dev/reports/fir-r02-rope10k-100k-20260906/README.md).
It includes 16 probe-fit chains, 4 probe-validation chains and 20,775 evaluation
chains. It does not include P-CORE datasets, model weights or training outputs.

Structures come from the **2024-02-28 RCSB Protein Data Bank snapshot**. Chain
selection and preprocessing remain unchanged: the benchmark is paper-faithful,
not claimed to be identical to the ESMC authors' unpublished chain selection.
The [evaluation contract](EVALUATION.md#contact-pl) defines probe fitting,
long-range contacts and confidence intervals.

PDB archive data are available under **CC0 1.0**, per the
[wwPDB usage policy](https://www.wwpdb.org/about/usage-policies).
Please acknowledge the PDB and original structure authors. Chain IDs remain in
`CONTACT_MANIFEST.jsonl`; PDB entry pages provide the associated publications.

> Berman, H. M. et al. The Protein Data Bank. *Nucleic Acids Research* 28,
> 235–242 (2000). [doi:10.1093/nar/28.1.235](https://doi.org/10.1093/nar/28.1.235).

The evaluator source is distributed under this repository's [MIT license](../LICENSE).
Its six source files are copied byte-for-byte from the recorded evaluator bundle;
`SOURCE_MANIFEST.json` retains their SHA-256 hashes. The contact manifest hash is
`c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3`.
The payload inventory hash is
`1b73f5f466420c8d0c74be452ebabe46af837482cee357674cad01d99e6f4b70`.

Setup verifies the archive checksum, source files, manifest, inventory and every
chain payload before reporting success. Both `evaluation/contact/` and
`evaluation/source/` live beneath `DATA_ROOT`. The
[packaging utility](../.dev/scripts/package_contact_evaluation.py) reproduces the
archive from the existing frozen dataset without altering its contents.
