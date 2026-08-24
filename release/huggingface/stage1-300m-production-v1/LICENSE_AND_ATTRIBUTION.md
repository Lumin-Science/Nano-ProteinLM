# License and attribution

Reviewed: 2026-08-23

## Distribution-license decision

Lumin Science distributes the rights it holds in this database compilation—its
selection, arrangement, decontamination ledger, binary packing, and original
release metadata—under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
Hugging Face metadata therefore uses `license: cc-by-sa-4.0`.

This license does **not** relicense third-party protein sequence records. Each
source arm remains governed by its direct-source terms below. The path-level
marking is an explicit third-party-content exception to the umbrella license.
Nothing in this file grants rights Lumin Science does not hold, overrides an
upstream license or terms of use, or implies endorsement by an upstream provider.

## UniRef90 arm

Paths: `uniref90/**`

Direct source: UniRef release 2023_02, obtained from the official UniProt release archive.
UniProt states that CC BY 4.0 applies to all copyrightable parts of its databases.

- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Source: [UniRef help](https://www.uniprot.org/help/uniref)
- License notice: [UniProt license and disclaimer](https://www.uniprot.org/help/license)
- Attribution: UniProt Consortium / UniRef
- Changes: representative extraction, sequence normalization, filtering, global exact
  deduplication with source membership retained, 70%-identity clustering, deterministic
  subset selection, evaluation decontamination, ESMC tokenization, and binary packing.

Users must provide appropriate attribution, retain the license notice, link to the
source where practicable, and indicate modifications. UniProt notes that patent or other
rights may apply to some records.

## MGnify arm

Paths: `mgnify/**`

Direct source: MGnify Protein Database release 2023_02 90%-identity cluster
representatives, obtained from the official EMBL-EBI FTP distribution.

- Governing terms: [EMBL-EBI Terms of Use](https://www.ebi.ac.uk/about/terms-of-use/)
- Resource: [MGnify Protein Database](https://www.ebi.ac.uk/metagenomics/proteins/)
- Licensing context: [Licensing of EMBL-EBI data resources](https://www.ebi.ac.uk/licencing/)
- Attribution: EMBL-EBI MGnify and the original data contributors
- Changes: sequence normalization, filtering, global exact deduplication with source
  membership retained, 70%-identity clustering, deterministic subset selection,
  evaluation decontamination, ESMC tokenization, and binary packing.

EMBL-EBI states that it imposes no additional restriction on use or redistribution
beyond restrictions supplied by original data owners and expects attribution in
accordance with good scientific practice. It also states that community contributors
remain data owners and that original data may be subject to third-party intellectual
property or biodiversity-related rights. Consequently, Lumin Science does **not** apply
CC BY, CC BY-SA, CC0, or another new license to `mgnify/**`; redistribution and use remain
subject to the EMBL-EBI Terms of Use and any applicable original-owner rights.

## OMG/IMG arm

Paths: `omg_img/**`

Direct source: the 959-shard `tattabio/OMG` Hugging Face distribution. This release keeps
only its numeric-accession JGI/IMG protein CDS rows and excludes its `ERZ...`
MGnify-origin rows.

- License declared by the direct source: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
- Source and license metadata: [`tattabio/OMG`](https://huggingface.co/datasets/tattabio/OMG)
- Underlying resource identified by OMG: [JGI IMG](https://img.jgi.doe.gov/)
- Attribution: Andre Cornman and the OMG authors; TattaBio; JGI/IMG and its original
  contributors
- Changes: JGI/IMG source selection, protein extraction, sequence normalization,
  filtering, global exact deduplication with source membership retained, 70%-identity
  clustering, deterministic subset selection, evaluation decontamination, ESMC
  tokenization, and binary packing.

Downstream sharing of this adapted arm must comply with CC BY-SA 4.0, including
attribution, modification notices, a license link, and ShareAlike where applicable.
Users should also review any record-level or underlying-source terms that apply to their
intended use.

Preferred OMG citation:

> Cornman, A., West-Roberts, J., Camargo, A. P., Roux, S., Beracochea, M., Mirdita, M.,
> Ovchinnikov, S., and Hwang, Y. (2024). *The OMG dataset: An Open MetaGenomic corpus
> for mixed-modality genomic language modeling*. bioRxiv.
> https://doi.org/10.1101/2024.08.14.607850

## Project-authored release metadata

Except for upstream material, factual identifiers, quotations, and linked license text,
the database compilation and release-specific documentation and metadata authored by
Lumin Science are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The source code in the GitHub repository is governed separately by its repository
`LICENSE` file. No software license changes the data terms above.

## No warranty

The data and metadata are provided as-is, without warranties. This provenance review is
an engineering record, not legal advice. Users are responsible for checking the current
upstream terms and any record-level obligations for their jurisdiction and intended use.
