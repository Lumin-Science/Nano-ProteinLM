import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import yaml

from nano_protein.evaluate import load_checkpoint
from nano_protein.model import count_parameters
from nano_protein.train import build_optimizer

root = Path(sys.argv[1])
method = sys.argv[2]
mode = sys.argv[3]
repo = Path('/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906-run')
expected_commit = '253c3ea442f2a1657eeb0f3ce2127ac0b1adfb25'
manifest = json.loads((repo/'configs/program2_h100_100k/manifest.json').read_text())
entry = next(x for x in manifest['runs'] if x['method'] == method)
source_config = repo/entry['config']
assert hashlib.sha256(source_config.read_bytes()).hexdigest() == entry['config_sha256']
expected = yaml.safe_load(source_config.read_text())
if mode == 'trial':
    expected.update(max_steps=200, schedule_steps=200, warmup_steps=50, walltime_seconds=1200)
else:
    assert mode == 'full'
actual_config = yaml.safe_load((root/'config.yaml').read_text())
assert actual_config == expected
completion = json.loads((root/'TRAINING_COMPLETE.json').read_text())
contract = json.loads((root/'run_contract.json').read_text())
assert completion['optimizer_steps'] == completion['target_optimizer_steps'] == expected['max_steps']
assert completion['schedule_optimizer_steps'] == expected['schedule_steps']
assert completion['sequences_seen'] == expected['max_steps'] * 1024
assert completion['stop_reason'] == 'max_steps'
assert completion['parameter_count'] == expected['expected_parameter_count']
assert contract['git_commit'] == expected_commit and not contract['git_dirty']
assert contract['world_size'] == 4 and contract['attention_kernel']['implementation'] == 'FlashAttention-3'
assert contract['attention_kernel']['revision'] == 'e29f138fc363b396e5d2706c8a5f6fa7d36f41e0'
assert contract['data_manifest_sha256'] == '43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab'
assert contract['config_sha256'] == hashlib.sha256((root/'config.yaml').read_bytes()).hexdigest()
rows = [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
rows = [row for row in rows if row.get('event') == 'train']
assert rows[-1]['optimizer_step'] == expected['max_steps']
for row in rows:
    assert all(math.isfinite(row[k]) for k in ('loss','gradient_norm','step_compute_seconds','learning_rate'))
    assert row['sequences_seen'] == row['optimizer_step'] * 1024
    assert row['attention_backend'] == 'flash3'
    if expected['balance_batches_across_ranks']:
        b = row['batch_balance']
        assert sum(b['rank_tokens_before']) == sum(b['rank_tokens_after'])
        assert max(b['rank_tokens_after']) <= max(b['rank_tokens_before'])
    if expected['training_loss_reduction'] == 'sqrt_mask_count':
        assert row['training_loss_reduction'] == 'sqrt_mask_count' and math.isfinite(row['objective_loss'])
assert abs(rows[-1]['learning_rate'] - 0.00045) < 1e-12
checkpoint = root/'checkpoint-final.pt'
with checkpoint.open('rb') as handle:
    digest = hashlib.file_digest(handle,'sha256').hexdigest()
assert digest == completion['final_checkpoint']['sha256']
receipt = {'status':'passed','method':method,'mode':mode,'config_sha256':contract['config_sha256'],
           'checkpoint_sha256':digest,'optimizer_steps':completion['optimizer_steps'],
           'model_tokens':completion['model_tokens'],'sequences_seen':completion['sequences_seen'],
           'parameter_count':completion['parameter_count'],'training_seconds':completion['training_seconds'],
           'peak_cuda_memory_bytes':completion['peak_cuda_memory_bytes'],
           'finite_logged_losses_and_gradients':True,'flash3_verified':True}
recent = rows[-11:]
receipt['measured_seconds_per_step'] = ((recent[-1]['training_seconds']-recent[0]['training_seconds']) /
                                      (recent[-1]['optimizer_step']-recent[0]['optimizer_step']))
if mode == 'trial':
    model, packet = load_checkpoint(checkpoint, torch.device('cuda',0))
    assert packet['train_config'] == expected
    assert count_parameters(model) == expected['expected_parameter_count']
    assert model.config.rotary_base == 10000 and model.config.ffn_hidden_dim == 2048
    assert model.config.transformer_norm == 'rmsnorm' and model.config.learned_residual_routing
    assert model.config.depth_scaled_residual_init
    assert (model.embedding.weight is model.head_out.weight) == expected['tie_word_embeddings']
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    optimizer = build_optimizer(model,expected)
    groups=[]; owners=[]
    for group in optimizer.param_groups:
        owners.extend(id(p) for p in group['params'])
        groups.append({'name':group['name'],'lr':expected['learning_rate']*group.get('lr_scale',1),
                       'weight_decay':group['weight_decay'],'tensors':len(group['params'])})
    assert len(owners) == len(set(owners))
    for name, current in optimizer.named_optimizers:
        saved=next(x['state_dict'] for x in packet['optimizer']['optimizers'] if x['name']==name)
        assert len(saved['state']) == sum(len(g['params']) for g in current.param_groups)
    tokens=torch.tensor([[0,4,5,6,7,8,9,10,11,12,2]],device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        result=model(tokens,torch.ones_like(tokens,dtype=torch.bool),output_attentions=True)
    assert torch.isfinite(result['logits']).all() and len(result['attentions'])==24
    assert all(a.shape==(1,12,11,11) and torch.isfinite(a).all() for a in result['attentions'])
    receipt.update(checkpoint_reload=True,all_parameters_finite=True,attention_feature_layout=True,
                   tied_weight_ownership_verified=True,optimizer_groups=groups)
(root/'TRAINING_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
