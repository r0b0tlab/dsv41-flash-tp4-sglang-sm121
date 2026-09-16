#!/usr/bin/env python3
"""Isolate BFCL results per fresh run; relocate only import-time locks.

The official evaluator and frozen subset source are unchanged. BFCL binds its
paths at import, so PROJECT_ROOT must never be switched after import. Only its
lock directory is placed in a fresh sibling to preserve the kit's empty-root
witness. All result/score/selection paths remain under the real run root.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys

ap=argparse.ArgumentParser()
ap.add_argument('root',type=pathlib.Path)
ap.add_argument('--inspect-root',action='store_true')
args=ap.parse_args()
root=args.root
if not root.is_absolute():raise SystemExit('fresh BFCL root must be absolute')
if root.exists():raise SystemExit('fresh BFCL root required; existing path retained')
locks=root.with_name(root.name+'.locks')
locks.mkdir(parents=True,exist_ok=False)
os.environ['BFCL_PROJECT_ROOT']=str(root)
import bfcl_eval.constants.eval_config as config
config.LOCK_DIR=locks
import bfcl_eval._llm_response_generation
assert config.PROJECT_ROOT==root and config.RESULT_PATH==root/'result' and config.SCORE_PATH==root/'score'
kit=pathlib.Path(os.environ.get('Q200_KIT',pathlib.Path.home()/'projects/r0b0bench/subsets/q200v2'))
sys.path.insert(0,str(kit/'scripts'))
import run_bfcl_hard20
assert run_bfcl_hard20.RESULT_PATH==root/'result'
assert run_bfcl_hard20.SCORE_PATH==root/'score'
assert run_bfcl_hard20.TEST_IDS_TO_GENERATE_PATH==root/'test_case_ids_to_generate.json'
files=sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
assert files==[], 'fresh result root unexpectedly populated during import'
receipt={'schema':'r0b0tlab.bfcl.path_isolation.v1','project_root':str(root),'result_path':str(config.RESULT_PATH),'score_path':str(config.SCORE_PATH),'lock_dir':str(locks),'root_files':files,'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'frozen_runner_sha256':hashlib.sha256(pathlib.Path(run_bfcl_hard20.__file__).read_bytes()).hexdigest()}
(root.with_name(root.name+'.path-isolation.json')).write_text(json.dumps(receipt,indent=2)+'\n')
if args.inspect_root:
    print(json.dumps(receipt));raise SystemExit(0)
if os.environ.get('Q200_ADMISSION_CONFIG'):
    from q200_support.admission_control import AdmissionCoordinator
    coordinator=AdmissionCoordinator.from_path(os.environ['Q200_ADMISSION_CONFIG'])
    original_query=run_bfcl_hard20.Q200OpenAICompletionsHandler._query_FC
    def admitted_query(self,inference_data):
        with coordinator.request(str(inference_data.get('q200_case_id','bfcl-turn'))):
            return original_query(self,inference_data)
    run_bfcl_hard20.Q200OpenAICompletionsHandler._query_FC=admitted_query
sys.argv=['run_bfcl_hard20.py','run']
raise SystemExit(run_bfcl_hard20.main())
