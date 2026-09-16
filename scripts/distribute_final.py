import concurrent.futures,json,pathlib,subprocess,sys
from final_runtime import ROOT,SSH,HOSTS,remote,save,require_cluster
require_cluster()
p=ROOT/'.hermes/closeout/build';image=(p/'image-id.txt').read_text().strip()

def transfer(rank):
    logfile=p/f'transfer-rank{rank}.log'
    with logfile.open('w') as log:
        producer=subprocess.Popen(['docker','save',image],stdout=subprocess.PIPE,stderr=log)
        consumer=subprocess.Popen([*SSH,HOSTS[rank],'docker load'],stdin=producer.stdout,stdout=log,stderr=log)
        producer.stdout.close()
        cr=consumer.wait();pr=producer.wait()
    if pr or cr:raise RuntimeError(f'transfer rank{rank} producer={pr} consumer={cr}')
    d=json.loads(remote(rank,['docker','image','inspect',image]))[0]
    assert d['Id']==image and d['Architecture']=='arm64'
    save(p/f'transfer-rank{rank}.json',{'image_id':d['Id'],'source':d['Config']['Labels']['org.opencontainers.image.revision'],'producer_rc':pr,'consumer_rc':cr})
    print('TRANSFER_VERIFIED',rank,image,flush=True)

with concurrent.futures.ThreadPoolExecutor(3) as ex:list(ex.map(transfer,(1,2,3)))
(p/'DISTRIBUTED').write_text(image+'\n')
