import importlib.util
import pathlib
import sys
import types

import pytest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'adapter'))
from sglang_patch.prefill_chunk_sizer import BudgetChunkSizer, patch_scheduler, settings


def test_budget_and_page_alignment():
    s=BudgetChunkSizer(300_000_000,256,2048)
    assert [s.predict(v) for v in [0,100000,200000,524288,1046462]] == [2048,2048,1280,512,256]
    for l in range(0,1048577,733):
        t=s.predict(l)
        assert t % 256 == 0 and 256<=t<=2048 and t*(l+t)<=300_000_000


@pytest.mark.parametrize('args',[(float('nan'),256,2048),(float('inf'),256,2048),(-1,256,2048),(3e8,0,2048),(3e8,100,2048),(3e8,256,1),(True,256,2048),(3e8,256,2048,0)])
def test_invalid_constructor(args):
    with pytest.raises(ValueError): BudgetChunkSizer(*args)


def test_impossible_floor_and_negative_history_fail():
    s=BudgetChunkSizer(3e8,256,2048)
    for history in [-1,10**7,True,1.25]:
        with pytest.raises(ValueError):s.predict(history)


def test_strict_environment():
    assert settings({})['budget']==3e8
    for env in [{'DSV41_CHUNK_BUDGET_TOKENS2':'nan'},{'DSV41_INDEXER_SCORE_BUDGET_MIB':'0'},{'DSV41_CHUNK_MIN':'True'}]:
        with pytest.raises(ValueError):settings(env)


def test_scheduler_install_idempotent_and_signature():
    class S:
        chunked_prefill_size=512
        page_size=256
        def maybe_init_dynamic_chunk_sizer(self):self.dynamic_chunk_sizer=None
    m=types.SimpleNamespace(Scheduler=S)
    cfg=settings({}); patch_scheduler(m,cfg); f=S.maybe_init_dynamic_chunk_sizer; patch_scheduler(m,cfg)
    assert S.maybe_init_dynamic_chunk_sizer is f
    s=S();s.maybe_init_dynamic_chunk_sizer();assert s.dynamic_chunk_sizer.predict(524288)==512
    s.chunked_prefill_size=2048
    with pytest.raises(RuntimeError): s.maybe_init_dynamic_chunk_sizer()


def test_no_heavy_import_for_module():
    import subprocess
    adapter=str(pathlib.Path(__file__).resolve().parents[1]/'adapter')
    p=subprocess.run([sys.executable,'-S','-c',f"import sys;sys.path.insert(0,{adapter!r});import sglang_patch.prefill_chunk_sizer;assert 'torch' not in sys.modules;assert 'sglang' not in sys.modules"],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
