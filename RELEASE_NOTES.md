# AutoResearch v1 — evaluation v3

The `autoresearch-v1` tag is a standalone clean starter with one root commit. Its plain ESMC model, AdamW training implementation and starting configurations are unchanged from `autoresearch-v0`. It contains no previous research recipes, findings or commit history. The research branch retains its own recipes and checkpoint compatibility.

## Evaluation defaults

- Search evaluates MLM loss and contact P@L on the same fixed 8,192 chains. The MLM reward gate selects candidates; P@L is reported alongside it.
- Scale-up reports exactly three metrics, each as mean and sample SD: P@L on 26,062 chains across five probe attempts; MLM on those chains across five fixed masks; MLM on the original 12,288 validation proteins across five fixed masks.
- Probe attempts retain the same 16/4 fit/validation split and vary pair sampling and logistic-fitting seeds. Mask attempts keep protein crops fixed. There is no separate single-mask headline score.
- Fresh setup requires the organizer-provided, checksummed v3 contact archive or the verified local recovery and prepares eleven MLM caches. The historical v2 source bundle still downloads automatically; the expanded v3 archive is not yet publicly hosted.
- Historical populations and results are preserved as historical records. Remeasure search baselines before using the new reward.

## Scientific qualifications

The 20 probe chains remain separate from the 26,062 scored chains. Homology exclusion of the additional 5,287 recovered chains is not established. An audit of available training prefixes found exact full-sequence matches for 30 added chains, including 29 evaluated-fragment matches. Four original chains have full-parent matches but no evaluated-fragment matches. These are availability checks, not evidence of which sequences a checkpoint sampled; the expanded population is not a newly established blind holdout.

## Setup

```bash
git clone --depth 1 --single-branch --no-tags --branch autoresearch-v1 \
  https://github.com/Lumin-Science/Nano-ProteinLM.git nano-protein-autoresearch
cd nano-protein-autoresearch
git remote remove origin
bash scripts/setup.sh --contact-v3-archive /path/to/contact-evaluation-v3.tar.gz
```

See `docs/EVALUATION.md` for exact populations, seeds, statistical definitions and data provenance. `RELEASE_MANIFEST.json` in the starter binds every released file by SHA-256. Dataset downloads remain separate from the source archive.
