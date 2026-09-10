import concurrent.futures,datetime,json,os,shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from nanoprotein.data import file_sha256
from nanoprotein.sharded_data import materialize_plan,validate_prepared_plan,validate_release_manifest
ROOT=Path('/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910')
OLD=Path('/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909')
def phase(name,**kw):
    (ROOT/'PREPARATION.json').write_text(json.dumps({'phase':name,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2)+'\n')
def main():
    spec=json.loads((ROOT/'DATA_PLAN.json').read_text());cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    mpath=Path(hf_hub_download(repo_id='LuminScience/LuminBench-Nano-ESMC',repo_type='dataset',revision=spec['release_revision'],filename='manifest.json',local_dir=cache))
    assert file_sha256(mpath)==spec['release_manifest_sha256']
    manifest=json.loads(mpath.read_text());validate_release_manifest(manifest)
    oldplan=json.loads((OLD/'cache/download-plan.json').read_text())
    plan={k:v for k,v in oldplan.items() if k not in ('paths','sources','total_training_samples','selected_records_including_validation','selected_residues_including_validation','selected_compressed_bytes_including_validation')}
    plan.update(sources={},paths=[],total_training_shards=399,planned_total_steps=400000,planned_global_batch=2048,source_resampling={'uniref90':'allow','mgnify':'error','omg_img':'allow'})
    shards=[]
    for name,s in spec['sources'].items():
        train=manifest['sources'][name]['train'][:s['selected_train_shards']];val=manifest['sources'][name]['validation']
        plan['sources'][name]={'train':train,'validation':val,'selected_unique_records':sum(x['records'] for x in train),'expected_draws':s['expected_draws_400k']}
        shards+=train+val
    plan['paths']=[x['path'] for x in shards]
    for key,field in [('selected_records_including_validation','records'),('selected_residues_including_validation','residues'),('selected_compressed_bytes_including_validation','bytes')]:plan[key]=sum(x[field] for x in shards)
    (ROOT/'download-plan.json').write_text(json.dumps(plan,sort_keys=True)+'\n')
    phase('downloading',additional_shards=188)
    def fetch(s):
        p=cache/s['path'];old=OLD/'cache'/s['path'];p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            if old.exists():os.link(old,p)
            else:hf_hub_download(repo_id='LuminScience/LuminBench-Nano-ESMC',repo_type='dataset',revision=spec['release_revision'],filename=s['path'],local_dir=cache)
        assert file_sha256(p)==s['sha256'],s['path']
        return s['path']
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for i,name in enumerate(pool.map(fetch,shards),1):
            if i%20==0:print(json.dumps({'verified_downloads':i,'total':len(shards),'last':name}),flush=True)
    (cache/'download-plan.json').write_text(json.dumps(plan,sort_keys=True)+'\n')
    (ROOT/'DOWNLOAD_VERIFIED.json').write_text(json.dumps({'status':'passed','plan':plan},indent=2)+'\n')
    local=Path(os.environ['SLURM_TMPDIR'])/'nano-nibi-setting3-400k-20260910'
    phase('materializing',data_root=str(local))
    proof=materialize_plan(plan,manifest,cache,local,workers=3)
    validate_prepared_plan(plan,local)
    oldmanifest=json.loads((OLD/'data/manifest.json').read_text());newmanifest=json.loads((local/'manifest.json').read_text())
    for source in newmanifest['sources']:assert newmanifest['sources'][source]['validation']==oldmanifest['sources'][source]['validation']
    phase('preserving_data',data_root=str(local))
    shutil.copytree(local,ROOT/'data');validate_prepared_plan(plan,ROOT/'data')
    (ROOT/'PRODUCTION_DATA_ROOT.txt').write_text(str(local)+'\n')
    (ROOT/'DATA_READY.json').write_text(json.dumps({'status':'passed','node_local_root':str(local),'durable_scratch_root':str(ROOT/'data'),'manifest':newmanifest,'verification':proof,'validation_identical_to_parent':True},indent=2)+'\n')
    phase('complete',status='passed',data_root=str(local));print('DATA_READY',flush=True)
if __name__=='__main__':main()
