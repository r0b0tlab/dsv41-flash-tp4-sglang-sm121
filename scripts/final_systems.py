"""Complete the four native SGLang bench_serving shape/concurrency cells."""
import argparse,json,pathlib,sys
from final_runtime import ROOT,MODEL
from final_eval import stage,verify_launch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--launch-dir',required=True);a=ap.parse_args();base=pathlib.Path(a.launch_dir).resolve()
    d=verify_launch(base)
    for shape,ni,no in [('short',128,128),('medium',2048,512)]:
        for c in [1,8]:
            name=f'bench-{shape}-c{c}';path=base/f'{name}.jsonl'
            if path.exists():raise RuntimeError('refusing append into historical benchmark')
            cmd=['docker','run','--rm','--network','host','--entrypoint','python3','-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=',
                 '-v',str(MODEL)+':/model:ro','-v',str(base)+':/out',
                 d['image_id'],'-m','sglang.bench_serving','--backend','sglang-oai-chat','--base-url','http://127.0.0.1:30000',
                 '--model','/model','--tokenizer','/model','--dataset-name','random','--random-input-len',str(ni),'--random-output-len',str(no),
                 '--num-prompts','20','--max-concurrency',str(c),'--seed','42','--extra-request-body',json.dumps({'chat_template_kwargs':{'thinking':False,'enable_thinking':False}}),
                 '--output-file','/out/'+path.name]
            stage(base,name,cmd)
            records=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            if len(records)!=1 or records[0]['completed']!=20:raise RuntimeError('incomplete or multiple bench records')
            print(name,records[0]['output_throughput'],flush=True)
if __name__=='__main__':main()
