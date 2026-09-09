"""Compare completed cumulative recipes using paired contact-chain resamples."""
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
root = repo / '.exps/fir-r02-rope10k-100k-20260906'
reference = repo / 'reports/fir-171m-100k-20260906'
methods = ['r02_rope10k', 'r04_batchbalance', 'r10_sqrtloss', 'r29_tied']
receipts = {}
values = {}
with (reference / 'contact-per-chain.tsv').open() as f:
    values['historical_r02_rope20k'] = {row['chain_id']: float(row['r02_p_at_l']) for row in csv.DictReader(f, delimiter='\t')}
receipts['historical_r02_rope20k'] = {
    'validation': json.loads((reference / 'r02/eval-validation/VALIDATION_MLM.json').read_text()),
    'p_at_l': json.loads((reference / 'r02/eval-p-at-l/P_AT_L.json').read_text())['p_at_l'],
    'training_seconds': json.loads((reference / 'r02/TRAINING_COMPLETE.json').read_text())['training_seconds'],
}
for method in methods:
    p = root / 'full' / method
    if not (p / 'RESULT_VERIFIED.json').exists():
        continue
    receipt = json.loads((p / 'RESULT_VERIFIED.json').read_text())
    assert receipt['status'] == 'passed' and receipt['optimizer_steps'] == 100000
    receipts[method] = receipt
    report = json.loads((p / 'eval-p-at-l/P_AT_L.json').read_text())
    rows = []
    for component in report['components']:
        rel = Path(component['path']).relative_to(f'/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906/full/{method}')
        rows.extend(json.loads((p / rel).read_text())['contact']['rows'])
    assert len(rows) == 20775
    values[method] = {row['chain_id']: row['precision_at_l'] for row in rows}

comparisons = []
sequence = ['historical_r02_rope20k'] + methods
changes = ['RoPE base 20000 to 10000', 'Add batch balancing', 'Add sqrt-mask-count training loss', 'Tie input/output embeddings']
for i, (baseline, candidate) in enumerate(zip(sequence, sequence[1:])):
    if baseline not in receipts or candidate not in receipts:
        continue
    b, c = receipts[baseline], receipts[candidate]
    assert values[baseline].keys() == values[candidate].keys()
    ids = sorted(values[baseline])
    differences = np.array([values[candidate][k] - values[baseline][k] for k in ids], dtype=np.float64)
    assert abs(differences.mean() - (c['p_at_l'] - b['p_at_l'])) < 1e-12
    rng = np.random.default_rng(20260820)
    means = np.empty(5000, dtype=np.float64)
    for start in range(0, 5000, 128):
        stop = min(start + 128, 5000)
        indices = rng.integers(0, len(ids), size=(stop - start, len(ids)))
        means[start:stop] = differences[indices].mean(axis=1)
    comparisons.append({
        'baseline': baseline, 'candidate': candidate, 'increment': changes[i],
        'validation_loss_delta': c['validation']['sequence_mean_nll'] - b['validation']['sequence_mean_nll'],
        'validation_loss_relative_delta_percent': 100 * (c['validation']['sequence_mean_nll'] / b['validation']['sequence_mean_nll'] - 1),
        'p_at_l_delta_percentage_points': 100 * (c['p_at_l'] - b['p_at_l']),
        'p_at_l_paired_chain_bootstrap_delta_95ci_percentage_points': (100 * np.quantile(means, [0.025, 0.975])).tolist(),
        'training_time_delta_seconds': c['training_seconds'] - b['training_seconds'],
        'training_time_relative_delta_percent': 100 * (c['training_seconds'] / b['training_seconds'] - 1),
        'paired_contact_chains': len(ids), 'bootstrap_replicates': 5000, 'bootstrap_seed': 20260820,
    })
packet = {
    'updated_utc': datetime.now(timezone.utc).isoformat(),
    'comparison_order': sequence,
    'comparisons': comparisons,
    'interpretation': 'Candidate minus baseline. Lower validation loss and training time are better; higher P@L is better. One matched training seed per recipe. Paired chain-bootstrap intervals condition on these trained checkpoints and do not measure training-seed variability or establish a reproducible training effect.',
}
(root / 'ADJACENT_COMPARISONS.json').write_text(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
