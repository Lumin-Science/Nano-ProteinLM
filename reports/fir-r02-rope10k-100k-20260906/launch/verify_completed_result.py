"""Independently audit one completed run against frozen source and reference data."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

SOURCE = '253c3ea442f2a1657eeb0f3ce2127ac0b1adfb25'
METHODS = ('r02_rope10k', 'r04_batchbalance', 'r10_sqrtloss', 'r29_tied')
repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('method', choices=METHODS)
parser.add_argument('--artifact-root', type=Path, default=repo / '.exps/fir-r02-rope10k-100k-20260906')
args = parser.parse_args()
ROOT = args.artifact_root.resolve()
method = args.method
p = ROOT / 'full' / method
read = lambda name: json.loads((p / name).read_text())
c = read('TRAINING_COMPLETE.json')
v = read('TRAINING_VERIFIED.json')
contract = read('run_contract.json')
mlm = read('eval-validation/VALIDATION_MLM.json')
ev = read('eval-validation/EVALUATION.json')
contact = read('eval-p-at-l/P_AT_L.json')
probe = read('eval-p-at-l/CONTACT_PROBE.json')
unc = read('eval-p-at-l/P_AT_L_UNCERTAINTY.json')
raw = subprocess.check_output(['git', 'show', f'{SOURCE}:configs/program2_h100_100k/{method}.yaml'], cwd=repo)
config = yaml.safe_load(raw)
gate = json.loads((ROOT / 'ALL_TRIALS_PASSED.json').read_text())
assert hashlib.sha256(raw).hexdigest() == gate['production_config_sha256'][method]
assert yaml.safe_load((p / 'config.yaml').read_text()) == config
assert (p / 'COMPLETE.txt').exists() and not (p / 'FAILED.txt').exists()
assert c['stop_reason'] == 'max_steps' and v['status'] == 'passed'
assert c['optimizer_steps'] == c['target_optimizer_steps'] == c['schedule_optimizer_steps'] == v['optimizer_steps'] == 100000
assert c['sequences_seen'] == v['sequences_seen'] == 102400000
assert c['model_tokens'] == v['model_tokens'] == 24200224761
assert c['parameter_count'] == config['expected_parameter_count']
assert contract['git_commit'] == SOURCE and not contract['git_dirty'] and contract['world_size'] == 4
assert contract['attention_kernel']['revision'] == 'e29f138fc363b396e5d2706c8a5f6fa7d36f41e0'
assert contract['data_manifest_sha256'] == '43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab'
assert contract['config_sha256'] == v['config_sha256'] == hashlib.sha256((p / 'config.yaml').read_bytes()).hexdigest()
checkpoint = c['final_checkpoint']['sha256']
assert checkpoint == v['checkpoint_sha256'] == ev['checkpoint_sha256'] == contact['checkpoint_sha256'] == probe['checkpoint_sha256'] == unc['checkpoint_sha256']
assert mlm == ev['validation_mlm'] and mlm['sequences'] == 4096 and mlm['masked_residues'] == 139963
assert all(math.isfinite(mlm[k]) for k in ('sequence_mean_nll', 'perplexity'))
old = repo / 'reports/fir-171m-100k-20260906/r02'
oldv = json.loads((old / 'eval-validation/VALIDATION_MLM.json').read_text())
oldprobe = json.loads((old / 'eval-p-at-l/CONTACT_PROBE.json').read_text())
assert mlm['source_counts'] == oldv['source_counts']
assert probe['dataset_manifest_sha256'] == 'c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3'
for key in ('probe_train_chain_ids', 'probe_validation_chain_ids', 'maximum_pairs_per_class', 'pair_sampling_seed', 'channels', 'protocol'):
    assert probe[key] == oldprobe[key], key
assert all(math.isfinite(x) for x in probe['coefficients']) and math.isfinite(probe['intercept'])
probe_sha = hashlib.sha256((p / 'eval-p-at-l/CONTACT_PROBE.json').read_bytes()).hexdigest()
rows, indices, digests = [], set(), {}
assert len(contact['components']) == 16
for component in contact['components']:
    rel = Path(component['path']).relative_to(f'/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906/full/{method}')
    f = p / rel
    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    assert digest == component['sha256']
    digests[str(rel)] = digest
    shard = json.loads(f.read_text())
    q = shard['contact']
    assert shard['checkpoint_sha256'] == checkpoint and q['probe_receipt_sha256'] == probe_sha
    assert q['shard_count'] == 16 and q['selection_total_chains'] == 20775
    assert q['selected_C'] == probe['selected_C'] == contact['selected_C']
    assert q['evaluation_chains'] == len(q['rows'])
    indices.add(q['shard_index'])
    rows.extend(q['rows'])
assert indices == set(range(16)) and len(rows) == len({row['chain_id'] for row in rows}) == contact['evaluation_chains'] == 20775
with (old.parent / 'contact-per-chain.tsv').open() as f:
    old_ids = {row['chain_id'] for row in csv.DictReader(f, delimiter='\t')}
assert {row['chain_id'] for row in rows} == old_ids
values = np.array([row['precision_at_l'] for row in rows], dtype=np.float64)
assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
assert abs(values.mean() - contact['p_at_l']) < 1e-12
# Match the frozen bootstrap algorithm: 5,000 chain resamples, 128 per chunk.
rng = np.random.default_rng(20260820)
means = np.empty(5000, dtype=np.float64)
for start in range(0, 5000, 128):
    stop = min(start + 128, 5000)
    samples = rng.integers(0, len(values), size=(stop - start, len(values)))
    means[start:stop] = values[samples].mean(axis=1)
ci = {'replicates': 5000, 'unit_count': len(values), 'confidence_interval_95': np.quantile(means, [0.025, 0.975]).tolist()}
assert ci == unc['uncertainty']
training = [json.loads(line) for line in (p / 'metrics.jsonl').read_text().splitlines()]
training = [row for row in training if row.get('event') == 'train']
assert training[-1]['optimizer_step'] == 100000
for row in training:
    assert all(math.isfinite(row[k]) for k in ('loss', 'gradient_norm', 'learning_rate'))
    assert row['attention_backend'] == 'flash3' and row['sequences_seen'] == row['optimizer_step'] * 1024
    if config['training_loss_reduction'] == 'sqrt_mask_count':
        assert math.isfinite(row['objective_loss']) and row['training_loss_reduction'] == 'sqrt_mask_count'
    if config['balance_batches_across_ranks']:
        b = row['batch_balance']
        assert sum(b['rank_tokens_before']) == sum(b['rank_tokens_after'])
assert abs(training[-1]['learning_rate'] - 0.00045) < 1e-12
receipt = {
    'verified_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'method': method,
    'checkpoint_sha256': checkpoint, 'source_commit': SOURCE, 'optimizer_steps': 100000,
    'sequences_seen': 102400000, 'model_tokens': c['model_tokens'], 'parameter_count': c['parameter_count'],
    'training_seconds': c['training_seconds'], 'validation': mlm, 'p_at_l': contact['p_at_l'],
    'p_at_l_uncertainty': ci, 'complete_shards': 16,
    'same_contact_chain_set_as_historical_comparison': True,
    'same_probe_split_and_fit_protocol_as_historical_comparison': True,
    'checkpoint_receipt_bindings_verified': True, 'component_sha256': digests,
    'bootstrap_independently_recomputed': True, 'finished_utc': (p / 'COMPLETE.txt').read_text().strip(),
}
(p / 'RESULT_VERIFIED.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'component_sha256'}, indent=2))
