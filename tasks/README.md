# Reusable AutoResearch task specifications

Start with the [five-section standard](../docs/AUTORESEARCH_TASK_STANDARD.md).
These YAML files describe experiments; pass the existing training/evaluation
configs to their runners, not these task specifications.

| File | Role |
|---|---|
| [task.schema.json](task.schema.json) | Common structural schema, version 1.0 |
| [protein-embedding.yaml](protein-embedding.yaml) | Complete protein question, baseline, search rule, paths/invariants and new 24.20B-token test specification |
| [OpenMM nonbonded](examples/openmm-nonbonded.yaml) | Mapping of the pinned neighbor/direct-space task into the same format |
| [OpenMM PME](examples/openmm-pme.yaml) | Mapping of the pinned reciprocal task into the same format |

To define another task, copy the closest example and replace the task identity,
question, established baseline, research protocol, boundaries and test protocol.
Declare the repeat unit and aggregation explicitly; do not inherit training
seeds or a token budget for a non-training task. Complete the `implementation`
status/gaps so readers can distinguish a defined rule from an enforced one.

Validate the YAML against the JSON Schema after loading it as JSON-compatible
data. Then check the scientific definitions against the actual runner. Before a
campaign, freeze a resolved manifest containing the task-spec digest, reference
commit/build, protected files, effective configs, external inputs/evaluator and
hardware/environment receipts. A contract version is not a substitute for those
runtime identities. No spec file launches work.

The OpenMM examples document remote main at commit `67b2c0f`. In particular,
its incumbent-improvement threshold and final regression policy remain
unspecified; the examples expose these gaps instead of assigning invented
thresholds to that project. Editing its protected task programs requires a
fresh matching baseline there. The OpenMM repository itself is unchanged.
