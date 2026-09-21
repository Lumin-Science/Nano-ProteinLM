# GitHub branch cleanup, September 21, 2026

At the owner's request, GitHub origin now retains only `main`. The five removed remote branches were `ar-260913-contact50`, `auto-research`, `autoresearch-171m`, `autoresearch-171m-val-loss` and `codex/esmc-300m-stage1-3d`. Local branch references, CCK branches, worktrees, uncommitted research changes, completed training artifacts and GPU allocations were retained. There were no open GitHub pull requests at inspection.

The default promotion and remote deletions were published together in an atomic Git push, with exact expected-commit leases for every changed reference and a verified fast-forward of main to `8e26910b2fd5a8910283576ac62d3800b8088f49`. The subsequent read of GitHub's branch references returned only main. [COMPLETED.json](COMPLETED.json) records that transaction; later documentation commits may advance main.

Before deletion, all six original origin heads and their complete reachable history were written to `outputs/branch-backups/20260921/origin-branches.bundle` in the local checkout. The bundle was restored into a separate bare repository and passed `git fsck --full --no-reflogs`. A second copy is at `/scratch/muchenli/Nano-ProteinLM/outputs/branch-backups/20260921/origin-branches.bundle` on CCK. Both copies have SHA-256 `143896f1338d9d061b04bf208f54cd51dcbe57bc4eab0c5ad01462d8c4e48b51`. The bundle is an external backup, not a Git-tracked binary. [BACKUP.json](BACKUP.json) records every original reference and commit.

To inspect the local backup from the repository root:

```bash
git bundle list-heads outputs/branch-backups/20260921/origin-branches.bundle
git bundle verify outputs/branch-backups/20260921/origin-branches.bundle
```

For example, to recover the contact-search history into a new local branch without creating a GitHub branch:

```bash
git fetch outputs/branch-backups/20260921/origin-branches.bundle \
  refs/remotes/origin/ar-260913-contact50:refs/heads/restored-contact50
```

The selected separate-Q/K/V implementation, tests and curated 100k-step evidence were imported onto main before the research branch reference was removed. See the [default promotion](../cck-contact-ablations-100k-20260919/DEFAULT_PROMOTION.md) and [benchmark-standard discussion draft](../../../docs/AUTORESEARCH_BENCHMARK_DRAFT.md).
