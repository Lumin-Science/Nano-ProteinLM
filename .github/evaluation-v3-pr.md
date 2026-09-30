## Changes

- Search measures MLM and P@L on the same fixed 8,192 chains. MLM remains the selection gate; P@L is reported alongside it.
- Scale-up reports three means and sample SDs: 26,062-chain P@L over five probe fits, 26,062-chain MLM over five fixed masks, and original 12,288-protein MLM over five fixed masks.
- Freeze crops and bind receipts to checkpoints, code, population IDs, seeds and prepared inputs. Historical component APIs require an explicit profile.
- Audit and update README, tasks, programs, setup/usage, data, protocol, evaluation and leaderboard docs. Preserve historical values with their original population labels.
- Integrate latest main (bb17aaa), including split-QKV defaults and numbered launch instructions.
- Publish autoresearch-v1 as a separate clean root with the unchanged plain ESMC model/training baseline and a SHA-256 file manifest.

## Verification

Eight v3/archive contract tests passed. A fresh archive install verified all 26,082 payload hashes, the 26,062/20 evaluation/probe split and unchanged 8,192 search IDs. Documentation links/anchors, Python compilation and shell syntax passed.

Earlier GPU verification reproduced the real-checkpoint 8,192-chain metrics exactly and exercised the full five-probe/five-mask pipeline with a tiny untrained model. The tiny-model run is a functional check, not a scientific result or a 171M timing estimate. Standalone starter compatibility tests are completed before tagging.

## Data and interpretation

Homology exclusion of the 5,287 added chains is unverified, and known exact training overlap is retained in docs and receipts. The expanded population is not a newly established blind holdout.

The verified portable v3 archive is prepared locally. Its upload was blocked by the read-only Hugging Face credential, so fresh setup explicitly requires an organizer-provided archive (`--contact-v3-archive`) or verified recovery (`--recovered-contact-pool`). Existing prepared data roots remain usable.

This PR targets main; it does not merge automatically. The one-time PR creation workflow is removed after opening the PR.
