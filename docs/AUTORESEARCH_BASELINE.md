# Karpathy-style sequential AutoResearch

Our sequential-search method proposes one change, measures it, keeps or discards it and continues from the retained recipe. It comes in two programs that differ only in how they decide what to keep: [karpathy_ar_reward_gate.md](../autoresearch/karpathy_ar_reward_gate.md) applies a fixed two-seed reward rule, and [karpathy_ar_agent_gate.md](../autoresearch/karpathy_ar_agent_gate.md) leaves the decision to the agent's reasoning. Running both on the same task and budget compares the two acceptance policies. The [AutoResearch protocol](AUTORESEARCH.md) fixes the scientific boundaries and budgets. Two rounds of this method, together with human effort, produced [nanop-best-171m-round2](leaderboard/nanop-best-171m-round2.md).

## Running the example loop

Follow the [launch steps](../README.md#launch-autoresearch) on your GPU compute node: clone the `autoresearch-v1` release without branch history, run setup, install the `ar-loop-n-sleep` skill, start your coding agent in tmux and give it the task and program.

| Method setting | Reward gate | Agent gate |
|---|---|---|
| Seeds | 42 for every run; 43 when seed 42 beats the incumbent's mean | 42 by default; more seeds when the agent decides |
| Candidate cost | One round if screened out, otherwise two | One round per run the agent chooses |
| Acceptance | Two-seed mean gain larger than the larger of the two seed SDs | The agent's reasoning about completed search measurements, recorded with its evidence |
| Task measurement | 20 minutes on four H100 GPUs with FA3; 4/3 H100 GPU-hours per round | Same |
| Search evaluation | MLM and P@L on the same fixed 8,192 chains; default selection remains MLM | Same |

The two baseline runs, seeds 42 and 43 of the starting recipe on the allocated GPUs, do not count toward the allowance. Under the reward gate each candidate then takes one or two rounds, so 72 rounds cover 36–72 candidates; under the agent gate, the count depends on how many rounds the agent spends on repeats and refinements. Each program lists the records to keep for every run.

## Run records

Record every budgeted training attempt, including failed experiments and repeated seeds, in `results.tsv`. Keep the hypothesis, code/configuration diff, final checkpoint digest, evaluation receipt, decision and consumed compute with the run. Append the reasoning to `research.log` and continue from the retained recipe. Baseline seeds 42 and 43 are outside the 72-round allowance; every subsequent training attempt consumes a round under the declared failure policy.

## Recipe benchmarks

The repository provides the plain [ESMC 171M reference](../configs/autoresearch/esmc-171m.yaml), [round 1](leaderboard/nanop-best-171m-round1.md) and [round 2](leaderboard/nanop-best-171m-round2.md). The [leaderboard](LEADERBOARD.md) compares their three-seed search checkpoints and completed scale-up checkpoints under the same [evaluation contract](EVALUATION.md). Recipe pages describe their implementation; checkpoint receipts identify the models behind each measurement.
