# License and attribution

Reviewed: 2026-08-23

Lumin Science distributes only the rights it holds in the database compilation—
selection, arrangement, the decontamination ledger, Parquet packing, and original
metadata—under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
This notice does not relicense third-party sequence records or identifiers.

## UniRef90 paths

`train/uniref90/**` and `validation/uniref90/**` derive from UniRef release
2023_02. UniProt applies [CC BY 4.0](https://www.uniprot.org/help/license) to
copyrightable database content. Attribute the UniProt Consortium / UniRef,
retain the notice, link to the source where practical, and state the processing
changes. UniProt notes that patents or other rights can apply to some records.

## MGnify paths

`train/mgnify/**` and `validation/mgnify/**` derive from the MGnify Protein
Database 2023_02 official FTP release. They remain governed by the
[EMBL-EBI Terms of Use](https://www.ebi.ac.uk/about/terms-of-use/) and any
applicable original-owner rights. Attribute EMBL-EBI MGnify and the original
contributors. Lumin Science does not apply a new CC license to these records.

## OMG/IMG paths

`train/omg_img/**` and `validation/omg_img/**` keep numeric-accession JGI/IMG
protein CDS rows from [`tattabio/OMG`](https://huggingface.co/datasets/tattabio/OMG)
and exclude its `ERZ...` MGnify-origin rows. The direct distribution declares
CC BY-SA 4.0. Attribute the OMG authors/TattaBio, JGI/IMG, and original
contributors; retain the license and modification notices and comply with
ShareAlike where applicable.

## Changes common to released paths

Sequences were extracted, normalized, quality filtered, globally exact
deduplicated with source membership retained, clustered at 70% identity,
decontaminated against all P@L and P-CORE evaluation splits, split into globally
disjoint train/validation sets, and packed into SHA-ordered Parquet shards.
