"""Render publication tables from the sanitized, verified results JSON."""
import hashlib,html,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=ROOT/'evidence/final/RESULTS.json'

def render(d):
    md=[];web=[]
    def para(s):md.append(s+'\n');web.append('<p>'+html.escape(s)+'</p>')
    def title(s,n=2):md.append('#'*n+' '+s+'\n');web.append(f'<h{n}>'+html.escape(s)+f'</h{n}>')
    def table(headers,rows):
        md.extend(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows],''])
        web.append('<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table>')
    title('DeepSeek-V4.1-Flash on 4×GB10 / SGLang TP4',1)
    para('r0b0tlab · @mr_r0b0t — official checkpoint, vision retained, local file-backed Engram lookup, static DSpark K5. Original adapter and reproducible runtime package; no model weights distributed.')
    para('Campaign state: '+d['status']+'. This is a measured profile, not a fastest-hardware or long-term-stability claim.')
    title('Final-image results')
    rows=[]
    q=d['quality'];text=q.get('text180',{})
    rows.append(['Q200v2',f"{q['closure']['correct_count']}/{q['closure']['total_count']} ({q['closure']['accuracy_pct']:.2f}%)" if q.get('closure') else 'PENDING — complete combined score not yet available','Text180 + official BFCL structural-hard20; thinking on, effort low'])
    if text.get('status')=='SCORED':rows.append(['Text180',f"{text['correct_count']}/{text['rows']}",'All rows normally terminated; independent manual review included'])
    for name,v in d['vision'].items():rows.append([name,f"{v['correct']}/{v['rows']} ({100*v['accuracy']:.2f}%)",'Thinking off; one worker; bounded CV-Bench first60 / full MMVP300'])
    mm=d['vision']['mmvp']['subsets']['mmvp']['paired'];rows.append(['MMVP paired',f"{mm['both_correct']}/{mm['pairs']}",'Both answers correct in each pair'])
    for name,v in d['retrieval'].items():rows.append(['NIAH '+name,v['status'],'Exactly one ordered two-key 33%/66% case; not a 25/50/90 ladder'])
    table(['Lane','Result','Scope'],rows)
    if q.get('closure'):table(['Q200 family','Correct','Total'],[[name,s['correct'],s['n']] for name,s in q['closure']['families'].items()])
    para('Manual review corrects erroneous frozen references for three cases. One answer is rejected for an incorrect additional continuous-time claim despite a correct discrete recurrence. See MANUAL-REVIEW-POLICY.json; scores are bound to unchanged response hashes. Historical overlay-v1 192/200 is not the score of this image.')
    title('Throughput: read each workload separately')
    para('Custom primary/counting rates use server usage tokens and real complete-client elapsed time. Thinking is off. Primary/counting values are medians of three measured repetitions after one excluded warmup. They include request prefill, not pure decode. Fixed-token synthetic rows are intentionally capped and are not semantic-answer quality scores.')
    table(['Workload','K5 aggregate output tok/s','K3 comparison tok/s'],[[name,f"{v['aggregate_tok_s']:.2f}",f"{d['k3_comparison'][name]['aggregate_tok_s']:.2f}"] for name,v in d['primary'].items()])
    para(d['selection_note'])
    table(['Counting workload','Aggregate output tok/s'],[[k,f"{v['aggregate_tok_s']:.2f}"] for k,v in d['counting'].items()])
    table(['Requested concurrency','Sustained observed','Completed','Errors','Aggregate output tok/s'],[[r['concurrency'],r['steady_running'],r['completed'],r['errors'],f"{r['aggregate_tok_s']:.2f}"] for r in d['ladder']['steps']])
    para('Load-counter caveat: '+d['ladder']['caveat'])
    title('Native SGLang bench_serving')
    if d['native_bench_serving']:
        table(['Cell','Completed','Sampled input/output maxima','Retokenized E2E tok/s','Harness nominal tok/s','Mean TTFT ms'],[[k,v['completed'],f"{v['random_input_len']}/{v['random_output_len']}",f"{v['retokenized_output_tok_s']:.2f}",f"{v['output_throughput']:.2f}",f"{v['mean_ttft_ms']:.1f}"] for k,v in d['native_bench_serving'].items()])
        para('Native cells use seed42 and random_range_ratio0: uniformly sampled lengths from1 to the displayed maxima, with ShareGPT-derived text, not fixed-length shapes. Harness nominal output counts can fall back to requested lengths if stream usage is absent; the decoded-response retokenized E2E rate is shown separately. Nominal input totals exclude any extra chat-template tokens. These rows are not interchangeable with the custom primary or quality workload rates.')
    else:para('PENDING. Historical bench-prod-tp4 copies are not independent control measurements and are not reused here.')
    title('Telemetry and reliability')
    tp=ROOT/'evidence/final/TELEMETRY.json'
    if tp.exists():
        telemetry=json.loads(tp.read_text());trows=[]
        for lane,v in telemetry['lanes'].items():
            for rank,r in v['thermal_by_rank'].items():
                temp=r.get('gpu_temperature_C');power=r.get('nvml_reported_power_W');mem=r.get('mem_available_bytes')
                trows.append([lane,rank,r['coverage'],f"{temp['mean']:.1f}/{temp['max']:.1f}" if temp else 'unavailable',f"{power['mean']:.1f}" if power else 'unavailable',f"{mem['min']/2**30:.2f}" if mem else 'unavailable'])
        table(['Lane','Rank','Coverage','GPU °C mean/max','NVML mean W','Min available GiB'],trows)
        erows=[]
        for lane,v in telemetry['lanes'].items():
            e=v.get('efficiency',{});rate=e.get('e2e_output_tok_s',e.get('output_throughput'))
            if rate is not None:erows.append([lane,f'{rate:.2f}',f"{v['lane_wall_seconds']:.1f}"])
        table(['Evaluation','Request-E2E output tok/s','Total lane wall seconds'],erows)
        para(telemetry['method'])
    para('A real NVRM allocation failure occurred during the first BFCL attempt after text180. All four guards stopped the runtime; those BFCL outputs are infrastructure-invalid, not scored model failures. Text180 was preserved unchanged and unfinished lanes were retried in a fresh identical-image/profile epoch. This recovery does not claim to repair the underlying allocation failure or establish indefinite service stability.')
    para('The first medium-c8 warmup later failed with an asynchronous CUDA illegal-memory-access error reported by rank0 NCCL. The three preceding native cells and completed Q200 were preserved. The missing cell is tested in another identical-image/profile epoch; the failure is retained and no originating kernel or stability repair is claimed.')
    para('Both512K execution attempts returned no answer after GPU allocation errors, including the identical-case replay in a cold epoch.512K is NOT retrieval-qualified; this is infrastructure failure, not a model retrieval miss. Host safety fixes require privileged kernel-journal visibility and10GiB resident-free admission and preserve primary errors. The separate1M C1 profile uses native allocator proactive reclamation at threshold0.6 with expansion off; the exact-image CUDA oracle verified the setting, but the full1M request also failed with a GPU allocation error before returning an answer. Neither512K nor1M is retrieval-qualified. The policy did not resolve the tested failure and is not transferred to production quality/performance.')
    para('Optional dedicated 2h mixed-workload soak: '+d['soak']['status']+'. Completed serial evaluations are not relabeled as that soak.')
    para('Final disposition: long-context qualification is BLOCKED on this frozen runtime. Both requested logical cases were attempted, but neither returned a gradeable answer. All four GPUs are released; the image, checkpoints and raw evidence are retained. No further image rebuild or node reboot was performed.')
    title('Runtime and reproduction')
    table(['Identity','Value'],[['Image config ID',d['image_id']],['Embedded local source',d['runtime_source']],['Upstream SGLang',d['engine_source']],['Prod profile SHA256',d['profile_sha256']],['Advertised prod window',d['context_length']]])
    if d.get('registry'):
        r=d['registry'];para('Verified registry reference: '+r['immutable_ref']);md.append('```bash\ndocker pull '+r['immutable_ref']+'\n```\n')
    else:para('New registry publication is pending. Do not treat an older overlay-v1 tag as this image.')
    para('Follow docs/REPRODUCIBILITY.md for private inventory, image verification, guarded launch/stop and serial evaluation. Public inventory is intentionally empty; no private host topology is embedded. The Dockerfile and adapter source match the immutable runtime as recorded in RUNTIME-PROVENANCE.json. Host-only publication changes are not a runtime rebuild.')
    para('Full machine-readable scores, timing/usage and source hashes: evidence/final/RESULTS.json, TEXT180-SCORES.json, VISION-SCORES.json, PERFORMANCE-ROWS.json, TELEMETRY.json and MANIFEST.sha256. Raw prompts/responses, host logs and credentials remain private. Historical phase directories are not current qualification.')
    para('Credit: DeepSeek, SGLang, FlashInfer, PyTorch/Triton, NVIDIA CUDA/CUTLASS/NCCL and the upstream benchmark authors. See THIRD_PARTY_NOTICES.md. Package code is MIT; model/base-image/data retain their own terms.')
    digest=hashlib.sha256(DATA.read_bytes()).hexdigest();para('Results JSON SHA256: '+digest)
    style='body{max-width:1160px;margin:40px auto;padding:0 22px;background:#101820;color:#e8eff5;font:16px/1.55 system-ui}h1,h2{color:#71e7db}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #334552;overflow-wrap:anywhere}th{background:#1b303b}p{overflow-wrap:anywhere}h2{margin-top:36px}'
    return '\n'.join(md),'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>r0b0tlab | DeepSeek-V4.1-Flash TP4</title><style>'+style+'</style><body>'+''.join(web)+'</body></html>\n'

if __name__=='__main__':
    data=json.loads(DATA.read_text());md,web=render(data)
    (ROOT/'README.md').write_text(md)
    (ROOT/'docs/report.html').write_text(web)
    print('Rendered README.md and docs/report.html from',DATA.relative_to(ROOT))
