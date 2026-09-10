import json,pathlib,os,datetime,shutil,yaml
from nanoprotein.sharded_data import materialize_plan,validate_prepared_plan
from nanoprotein.data_budget import data_coverage
from nanoprotein.train import validate_data_manifest

def main():
 root=pathlib.Path("/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909")
 repo=pathlib.Path(str(root)+"-run")
 plan=json.loads((root/"DOWNLOAD_VERIFIED.json").read_text())["plan"]
 release=json.loads((root/"cache/manifest.json").read_text())
 local=pathlib.Path(os.environ["SLURM_TMPDIR"])/"nano-nibi-paired-unique-20260909"
 (root/"PREPARATION.json").write_text(json.dumps({"phase":"materializing","started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"data_root":str(local)},indent=2)+"\n")
 receipt=materialize_plan(plan,release,root/"cache",local,workers=3)
 validate_prepared_plan(plan,local)
 manifest=validate_data_manifest(local)
 old=json.loads((pathlib.Path(os.environ["SLURM_TMPDIR"])/"nano-esmc-data-bd38448d/manifest.json").read_text())
 for source in manifest["sources"]:
  assert manifest["sources"][source]["validation"]==old["sources"][source]["validation"]
 coverage={}
 for name in ("baseline","setting3"):
  config=yaml.safe_load((repo/f"configs/esmc-171m-{name}-nibi-unique-b2048-100k.yaml").read_text())
  coverage[name]=data_coverage(config,manifest,world_size=4)
  assert coverage[name]["status"]=="passed"
 (root/"PREPARATION.json").write_text(json.dumps({"phase":"copying_durable_data","local_verified":True,"data_root":str(local)},indent=2)+"\n")
 shutil.copytree(local,root/"data")
 validate_prepared_plan(plan,root/"data")
 (root/"PRODUCTION_DATA_ROOT.txt").write_text(str(local)+"\n")
 result={"status":"passed","source_commit":(root/"SOURCE_COMMIT.txt").read_text().strip(),"time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"node_local_root":str(local),"durable_scratch_root":str(root/"data"),"validation_identical_to_previous_runs":True,"manifest":manifest,"verification":receipt,"coverage":coverage}
 (root/"DATA_READY.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({"status":"passed","data_root":str(local),"training_records":sum(v["train"]["records"] for v in manifest["sources"].values())}),flush=True)

if __name__=="__main__":main()
