"""Wait for one exact launch's readiness, then run its frozen primary lanes."""
import argparse,json,pathlib,subprocess,sys,time
from final_runtime import ROOT,remote,save
ap=argparse.ArgumentParser();ap.add_argument('--launch-dir',required=True);a=ap.parse_args();base=pathlib.Path(a.launch_dir).resolve()
deadline=time.monotonic()+3100
while not (base/'READY.json').exists():
    if time.monotonic()>deadline:raise SystemExit('specific launch readiness deadline')
    if (base/'launch.json').exists():
        record=json.loads((base/'launch.json').read_text())
        for row in record['ranks']:
            d=json.loads(remote(row['rank'],['docker','inspect',row['container_id']]))[0]
            if not d['State']['Running']:
                save(base/'PRIMARY_BLOCKED.json',{'reason':'owned rank stopped before readiness','rank':row['rank'],'image_id':d['Image']});raise SystemExit(2)
    time.sleep(10)
rc=subprocess.run([sys.executable,str(ROOT/'scripts/final_eval.py'),'primary','--launch-dir',str(base)]).returncode
save(base/'PRIMARY_OWNER_EXIT.json',{'exit_code':rc,'time':time.time()});raise SystemExit(rc)
