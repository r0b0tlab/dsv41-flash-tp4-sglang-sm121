import pathlib,sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts/q200_support'))
import guard_unified_memory as g

def test_kernel_counter_requires_privileged_complete_journal(monkeypatch):
    def run(cmd,**kw):
        assert cmd[:2]==['sudo','-n']
        return SimpleNamespace(returncode=0,stdout='Linux boot\nNV_ERR_NO_MEMORY\n',stderr='')
    monkeypatch.setattr(g.subprocess,'run',run)
    assert g.read_nvrm_count()==1

def test_unprivileged_success_is_not_a_fallback(monkeypatch):
    def run(cmd,**kw):
        return SimpleNamespace(returncode=1,stdout='',stderr='not permitted') if cmd[0]=='sudo' else SimpleNamespace(returncode=0,stdout='Linux boot',stderr='')
    monkeypatch.setattr(g.subprocess,'run',run)
    with pytest.raises(g.GuardError):g.read_nvrm_count()

def test_empty_journal_is_not_zero_errors(monkeypatch):
    monkeypatch.setattr(g.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='',stderr=''))
    with pytest.raises(g.GuardError):g.read_nvrm_count()
