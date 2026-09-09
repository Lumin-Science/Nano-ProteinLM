import json,pathlib,datetime
from nanoprotein.sharded_data import fetch_release_plan
root=pathlib.Path("/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909")
plan,path=fetch_release_plan(repo_id="LuminScience/LuminBench-Nano-ESMC",revision="bd38448d50d8f426d7b9bd4410b53159ea001259",cache_root=root/"cache",total_training_samples=206848000,weights={"uniref90":36.,"mgnify":11.,"omg_img":54.},download_workers=12)
assert plan["release_manifest_sha256"]=="fe1ac0657085ab19fe6f56786006e9eb004ca66bc6c5b81dfd8e6bc3dcfda6ff"
(root/"DOWNLOAD_VERIFIED.json").write_text(json.dumps({"status":"passed","time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"plan":plan},indent=2)+"\n")
