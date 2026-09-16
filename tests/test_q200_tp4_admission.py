import json
import sys
from pathlib import Path

import pytest

SUPPORT = Path(__file__).resolve().parents[1] / 'scripts' / 'q200_support'
sys.path.insert(0, str(SUPPORT))
import admission_control as ac
import guard_unified_memory as gm


def config(tmp_path, ranks=('0', '1', '2', '3')):
    d = dict(schema=ac.CONFIG_SCHEMA, epoch='test', candidate_id='dsv41-test',
             candidate_source_sha='a'*40, profile_sha256='b'*64,
             image_id='sha256:'+'c'*64, lease_path=str(tmp_path/'lease'),
             ranks=[dict(rank=r, host=None, admission_state_path=str(tmp_path/r/'state'),
                         request_state_path=str(tmp_path/r/'request')) for r in ranks])
    p=tmp_path/'config.json'; p.write_text(json.dumps(d)); return p


def test_four_rank_config(tmp_path):
    c=ac.load_config(config(tmp_path))
    assert [r.rank for r in c.ranks] == ['0','1','2','3']


@pytest.mark.parametrize('ranks', [('0','1'),('0','1','2'),('0','1','2','2'),('0','1','2','4')])
def test_partial_or_duplicate_rank_config_refused(tmp_path,ranks):
    with pytest.raises(ac.AdmissionError): ac.load_config(config(tmp_path,ranks))


def test_guard_owns_rank_three_and_ring_peer():
    common=dict(candidate_id='dsv41-test', candidate_source_sha='a'*40,
                epoch='test', profile_sha256='b'*64, image_id='sha256:'+'c'*64)
    gm.validate_ownership_arguments(**common, rank='3', container_id='d'*64,
        owner_nonce=gm.ownership_nonce(**common,rank='3'), peer_rank='0',
        peer_candidate_id=common['candidate_id'],peer_candidate_source_sha=common['candidate_source_sha'],
        peer_profile_sha256=common['profile_sha256'],peer_image_id=common['image_id'],peer_container_id='e'*64,
        peer_owner_nonce=gm.ownership_nonce(**common,rank='0'))


def test_ack_waits_for_all_four(tmp_path, monkeypatch):
    c=ac.load_config(config(tmp_path)); coord=ac.AdmissionCoordinator(c)
    monkeypatch.setattr(coord,'_read_bytes',lambda endpoint,path: b'{}' if endpoint.rank!='3' else b'bad')
    monkeypatch.setattr(ac,'validate_guard_state',lambda *a,**k: {'ok':True})
    clock=iter([0,0,1000]); monkeypatch.setattr(ac.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(ac.time,'sleep',lambda _:None)
    with pytest.raises(ac.AdmissionError, match='acknowledgement'):
        coord._wait_ack(active=True,lease_id='d'*64,require_admit=True)


def test_ssh_is_pinned():
    ep=ac.RankEndpoint('1',Path('/state'),Path('/request'),host='node',ssh_identity_file=Path('/key'))
    args=ac.AdmissionCoordinator._ssh_args(ep)
    assert 'StrictHostKeyChecking=yes' in args
    assert any('known_hosts_crs812_fabric' in v for v in args)


def test_loading_swap_exception_is_narrow():
    kw=dict(mem_available=32*2**30,admission_floor=16*2**30,low_samples=0,
            swap_growth=800*2**20,state={'Running':True,'OOMKilled':False},
            psi_hard_enabled=False,request_active=False,allow_load_swap=True)
    assert gm.evaluate_guard(**kw)['status']=='GREEN'
    for change in [dict(request_active=True),dict(psi_hard_enabled=True),dict(mem_available=15*2**30),dict(nvrm_count=2,baseline_nvrm=1),dict(events={'oom':1},baseline_events={'oom':0})]:
        assert gm.evaluate_guard(**(kw|change))['status']=='HARD_ABORT'


def test_psi_admission_only_preserves_fatal_memory_gates():
    kw=dict(mem_available=21*2**30,admission_floor=16*2**30,low_samples=0,
            psi_full_avg10=6.83,state={'Running':True,'OOMKilled':False},
            request_active=True,pressure_admission_only=True)
    d=gm.evaluate_guard(**kw)
    assert d['status']=='WARNING' and d['admission']=='NOT_ADMITTED'
    assert not d['peer_stop'] and d['action']=='DO_NOT_ADMIT_NEXT'
    for change in [dict(mem_available=7*2**30,low_samples=20),dict(nvrm_count=2,baseline_nvrm=1),dict(events={'oom':1},baseline_events={'oom':0})]:
        assert gm.evaluate_guard(**(kw|change))['status']=='HARD_ABORT'
