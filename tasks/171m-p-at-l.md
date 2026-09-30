# Paired search evaluation: compatibility task name

Use [171m-validation-loss.md](171m-validation-loss.md) for the current search contract. Both task entry points now evaluate MLM and P@L on the same fixed 8,192 chains, select by MLM, and report P@L alongside it. This older filename remains a compatibility alias; it does not introduce a P@L selection gate.

```bash
bash tasks/171m-p-at-l_ar.sh configs/autoresearch/esmc-171m.yaml experiment-001 42
```

Training budgets, replication policy, model/data boundaries and final evaluation are exactly those of the validation-loss task. The historical P@L-primary task and its results retain their original meanings in older checkouts. Current final reporting uses five probe attempts and five masks as defined in [EVALUATION.md](../docs/EVALUATION.md).
