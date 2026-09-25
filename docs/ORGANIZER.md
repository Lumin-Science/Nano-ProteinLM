# Organizing a comparison

Provision every participant from the same `autoresearch-v0` release tag and record its commit. The documented clone downloads only that root commit, and the remote is removed afterwards. Give each agent a fresh context, prepared data and the same compute allowance. Keep prior experiments, checkpoint stores, private notes, shell history and other checkouts outside the agent workspace.

Prepare dependencies and verified data before agent access. For a closed comparison, restrict network access after provisioning; an independent release commit and written rules do not block online access. Declare the external-information policy, agent/model version and proposal-generation budget before the comparison, and use the same policy for every method.

Keep an organizer-owned copy of the protocol, evaluator, evaluation assets and release manifest outside the agent's writable workspace. Run candidate training and model code in an isolated process with only the required data and permissions. Score submitted checkpoints using the trusted evaluation procedure, and independently check architecture changes, data boundaries, parameter counts and the compute ledger. A manifest inside an editable workspace is a reference, not an enforcement mechanism.

The release excludes previous research findings from its code and commit ancestry. It does not by itself prevent metric tampering or adaptation to repeatedly observed validation results. The published final evaluation shares the search evaluation assets; any blind generalization study needs a separately declared held-out set.
