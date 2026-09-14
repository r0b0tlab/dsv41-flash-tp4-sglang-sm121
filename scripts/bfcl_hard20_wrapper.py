#!/usr/bin/env python3
"""Launch wrapper: pre-import bfcl_eval with a scratch BFCL_PROJECT_ROOT so its
import-time lock/result/score dirs land there, then run the frozen hard20
runner with the real (pristine) root. No frozen code modified."""
import os
import sys

scratch = "/tmp/bfcl-scratch-root"
os.environ["BFCL_PROJECT_ROOT"] = scratch
import pathlib
pathlib.Path(scratch, ".file_locks").mkdir(parents=True, exist_ok=True)
import bfcl_eval._llm_response_generation  # noqa: F401  (locks instantiate here)

real_root = sys.argv[1]
os.environ["BFCL_PROJECT_ROOT"] = real_root
sys.argv = ["run_bfcl_hard20.py", "run"]

sys.path.insert(0, "/home/r0b0tdgx/projects/r0b0bench/subsets/q200v2/scripts")
import run_bfcl_hard20
raise SystemExit(run_bfcl_hard20.main())
