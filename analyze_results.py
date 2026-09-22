#!/usr/bin/env python3
"""Independent, CPU-only audit of the frozen v4 LoRA experiment exports."""
import argparse, hashlib, json, math, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def readj(path): return json.loads(Path(path).read_text())
def writej(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def expected(protocol):
    ans = []
    for e in protocol['budget_epochs']:
        for s in protocol['primary_seeds']:
            for m in protocol['primary_methods']: ans.append((m, e, s, False, 'primary'))
    for x in protocol['diagnostic_ablations']: ans.append((x['method'], x['epochs'], x['seed'], False, 'diagnostic'))
    x = protocol['planned_repeat']; ans.append((x['method'], x['epochs'], x['seed'], True, 'repeat'))
    return ans

def metric(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tp = int(((y == 1) & (p == 1)).sum())
    n = len(y); prec = tp / (tp + fp) if tp + fp else 0.; recp = tp / (tp + fn) if tp + fn else 0.
    recn = tn / (tn + fp) if tn + fp else 0.; f1 = 2 * prec * recp / (prec + recp) if prec + recp else 0.
    den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); mcc = ((tp*tn-fp*fn)/den) if den else 0.
    return {'n': n, 'confusion_matrix': [[tn, fp], [fn, tp]], 'accuracy': (tp+tn)/n if n else 0.,
            'binary_f1': f1, 'balanced_accuracy': (recp+recn)/2, 'mcc': mcc,
            'negative_recall': recn, 'positive_recall': recp, 'error_rate': 1-(tp+tn)/n if n else 0.}

def load_predictions(path, label_map, expected_ids=None):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    issues, ids, y, p, seen = [], [], [], [], set()
    for r in rows:
        try:
            i, lab, pred, logits = int(r['idx']), int(r['label']), int(r['prediction']), r['logits']
            if lab not in (0, 1): issues.append(f'nonbinary_label:{i}')
            if pred not in (0, 1): issues.append(f'nonbinary_prediction:{i}')
            if not isinstance(logits, list) or len(logits) != 2 or not np.isfinite(np.asarray(logits, dtype=float)).all():
                issues.append(f'invalid_logits:{i}'); continue
            arg = int(np.argmax(np.asarray(logits, dtype=float)))
            if i in seen: issues.append(f'duplicate_idx:{i}')
            seen.add(i)
            if i not in label_map: issues.append(f'unknown_idx:{i}')
            elif lab != int(label_map[i]): issues.append(f'label_mismatch:{i}')
            if pred != arg: issues.append(f'argmax_mismatch:{i}')
            ids.append(i); y.append(lab); p.append(pred)
        except (KeyError, TypeError, ValueError): issues.append('malformed_prediction')
    if expected_ids is not None and set(ids) != set(expected_ids): issues.append('id_set_not_exact_expected_split')
    return rows, np.asarray(ids), np.asarray(y), np.asarray(p), issues

def norm_tokens(x): return set(re.findall(r"\w+", str(x).lower(), flags=re.UNICODE))
def strata(frame, ids, y, p):
    byid = frame.set_index('idx'); out = {}
    def add(name, mask): out[name] = metric(y[mask], p[mask])
    add('gold_0', y == 0); add('gold_1', y == 1)
    jac = []
    for i in ids:
        r = byid.loc[int(i)]; a, b = norm_tokens(r['sentence1']), norm_tokens(r['sentence2'])
        jac.append(len(a & b) / len(a | b) if a | b else 1.)
    jac = np.asarray(jac)
    for name, mask in [('jaccard_[0,.5)', jac < .5), ('jaccard_[.5,.8)', (jac >= .5)&(jac < .8)), ('jaccard_[.8,1]', jac >= .8)]: add(name, mask)
    return out

def add_token_lengths(strata_out, frame, ids, y, p, asset_dir):
    """Use the pinned corrected tokenizer; never approximate lengths by words."""
    from tokenization import load_tokenizer
    selected = frame.set_index('idx').loc[list(map(int, ids))]
    tok = load_tokenizer(asset_dir / 'model')
    # Length bands describe raw corrected-tokenizer sequence length, not a clipped 128-token input.
    enc = tok(selected['sentence1'].tolist(), selected['sentence2'].tolist(), truncation=False, padding=False)
    lengths = np.asarray([len(x) for x in enc['input_ids']])
    for name, mask in [('length_<=64', lengths <= 64), ('length_65_128', (lengths >= 65)&(lengths <= 128)), ('length_>128', lengths > 128)]: strata_out[name] = metric(y[mask], p[mask])
    strata_out['length_tokenized_provenance'] = {'status':'PINNED_CORRECT_TOKENIZER_NO_TRUNCATION','min':int(lengths.min()),'max':int(lengths.max())}

def find_paws_parquet(paws_dir):
    hits = list(paws_dir.rglob('test.parquet')) + list(paws_dir.rglob('*test*.parquet'))
    return hits[0] if hits else None

def paws_audit(paws_dir, results_dir, protocol, runtime_root=None):
    data_path = find_paws_parquet(paws_dir)
    if data_path is None: return {'status':'INCOMPLETE','reason':'PAWS test.parquet missing'}
    frame = pd.read_parquet(data_path)
    if 'idx' not in frame.columns:
        if 'id' not in frame.columns: return {'status':'INCOMPLETE','reason':'PAWS parquet lacks source id column; dataframe row index is prohibited'}
        frame=frame.copy(); frame['idx']=frame['id']
    if 'label' not in frame.columns: return {'status':'INCOMPLETE','reason':'PAWS parquet lacks label'}
    if len(frame)!=8000 or frame.idx.duplicated().any() or set(frame.label.unique()) != {0,1}: return {'status':'INCOMPLETE','reason':'PAWS requires exactly 8000 unique source ids and both binary labels'}
    labels=dict(zip(frame.idx,frame.label)); expected_names=[f'{m}-e12-seed{s}' for m in ['full','lora_r8'] for s in protocol['primary_seeds']]
    found=[]; missing=[]; failures=[]
    stress_root=results_dir.parent/'stress'
    lock_path=stress_root/'locked_model_selection.json'
    if not lock_path.exists(): return {'status':'INCOMPLETE','reason':'locked_model_selection.json missing','test_parquet':str(data_path)}
    lock_rows=readj(lock_path); locked={x.get('run'):x for x in lock_rows}
    if len(lock_rows)!=len(expected_names) or set(locked)!=set(expected_names): return {'status':'INCOMPLETE','reason':'locked_model_selection.json does not contain exactly the six declared runs','test_parquet':str(data_path)}
    for name in expected_names:
        d=stress_root/name
        if not d.exists(): missing.append(name); continue
        pred=d/'predictions.jsonl'; summary=d/'summary.json'
        if not pred.exists() or not summary.exists(): missing.append(name); continue
        rows,ids,y,p,issues=load_predictions(pred,labels,labels.keys()); met=metric(y,p); declared=readj(summary)
        six=['accuracy','binary_f1','balanced_accuracy','mcc','negative_recall','positive_recall']
        metric_missing=[k for k in six if k not in declared.get('metrics',{})]
        source=results_dir/name/'summary.json'; source_summary=readj(source) if source.exists() else {}; lock=locked.get(name,{})
        checks={'rows':len(rows),'expected_rows':8000,'issues':issues,'prediction_sha256':{'expected':declared.get('prediction_sha256'),'observed':sha(pred)},'data_sha256':{'expected':declared.get('data_sha256'),'observed':sha(data_path)},'declared_metric_differences':{k:abs(met[k]-declared['metrics'][k]) for k in six if k in declared.get('metrics',{})},'metric_missing':metric_missing,'no_training_declared':declared.get('no_paws_training') is True,'checkpoint_locked_selection':bool(lock) and declared.get('checkpoint_sha256')==lock.get('checkpoint_sha256')==source_summary.get('best_checkpoint_sha256') and declared.get('selected_mrpc_dev_epoch')==lock.get('selected_mrpc_dev_epoch')==source_summary.get('selected_epoch')}
        ok=len(rows)==8000 and not issues and not metric_missing and checks['prediction_sha256']['expected']==checks['prediction_sha256']['observed'] and checks['data_sha256']['expected']==checks['data_sha256']['observed'] and checks['no_training_declared'] and checks['checkpoint_locked_selection'] and all(v<1e-12 for v in checks['declared_metric_differences'].values())
        found.append({'name':name,'metrics':met,'checks':checks,'passed':ok})
        if not ok: failures.append(name)
    return {'status':'COMPLETE' if len(frame)==8000 and not missing and not failures else 'INCOMPLETE','test_parquet':str(data_path),'test_rows':len(frame),'expected_runs':expected_names,'missing_runs':missing,'failed_runs':failures,'records':found}

def bootstrap(a_y, a_p, b_y, b_p, seed, draws=2000):
    a_y,a_p,b_y,b_p=(np.asarray(x,dtype=int) for x in (a_y,a_p,b_y,b_p))
    rng = np.random.default_rng(seed); n = len(a_y); keys = ['accuracy','binary_f1','balanced_accuracy','mcc','negative_recall','positive_recall']
    d = {k: [] for k in keys}
    for _ in range(draws):
        ix = rng.integers(0, n, n); am, bm = metric(a_y[ix], a_p[ix]), metric(b_y[ix], b_p[ix])
        for k in keys: d[k].append(am[k]-bm[k])
    observed_a, observed_b = metric(a_y,a_p), metric(b_y,b_p)
    return {k: {'observed_difference':float(observed_a[k]-observed_b[k]),'bootstrap_mean_difference': float(np.mean(v)), 'ci95_percentile': [float(np.quantile(v,.025)), float(np.quantile(v,.975))]} for k,v in d.items()}

def verify_source(run, asset_dir, runtime_root=None):
    env = readj(run/'environment.json'); checks = []
    for rel, wanted in env.get('source_sha256', {}).items():
        p = ROOT/rel; checks.append({'path':rel,'expected':wanted,'observed':sha(p) if p.exists() else None,'match':p.exists() and sha(p)==wanted})
    am = asset_dir/'asset_manifest.json'; checks.append({'path':'asset_manifest.json','expected':env.get('asset_manifest_sha256'),'observed':sha(am) if am.exists() else None,'match':am.exists() and sha(am)==env.get('asset_manifest_sha256')})
    if am.exists():
        for item in readj(am).get('assets',[]):
            rel=item.get('relative_path') or item.get('source_path'); p=(asset_dir/rel) if item.get('relative_path') else (ROOT/rel if rel else None)
            checks.append({'path':'asset:'+str(rel),'expected':item.get('sha256'),'observed':sha(p) if p and p.exists() else None,'match':bool(p and p.exists() and sha(p)==item.get('sha256'))})
    frozen=readj(ROOT/'pre_training_manifest.json').get('sha256',{})
    for rel,wanted in frozen.items():
        p=ROOT/rel; checks.append({'path':'prefreeze:'+rel,'expected':wanted,'observed':sha(p) if p.exists() else None,'match':bool(p.exists() and sha(p)==wanted)})
    summary = readj(run/'summary.json'); cp = run/'best.safetensors'
    if cp.exists(): checks.append({'path':'best.safetensors','expected':summary.get('best_checkpoint_sha256'),'observed':sha(cp),'match':sha(cp)==summary.get('best_checkpoint_sha256')})
    else:
        runtime_cp=(runtime_root/'runs'/run.name/'best.safetensors') if runtime_root else None
        note=run/'checkpoint_note.json'
        note_ok=note.exists() and readj(note).get('sha256')==summary.get('best_checkpoint_sha256') and readj(note).get('restoration_verified') is True
        checks.append({'path':'best.safetensors','status':'NOT_EXPORTED', 'runtime_path':str(runtime_cp) if runtime_cp else None,'expected':summary.get('best_checkpoint_sha256'),'observed':sha(runtime_cp) if runtime_cp and runtime_cp.exists() else None,'match':bool(runtime_cp and runtime_cp.exists() and sha(runtime_cp)==summary.get('best_checkpoint_sha256') and note_ok)})
    checks.append({'path':'summary.protocol_sha256','expected':sha(ROOT/'protocol.json'),'observed':summary.get('protocol_sha256'),'match':summary.get('protocol_sha256')==sha(ROOT/'protocol.json')})
    checks.append({'path':'predictions.jsonl','expected':summary.get('prediction_sha256'),'observed':sha(run/'predictions.jsonl'),'match':summary.get('prediction_sha256')==sha(run/'predictions.jsonl')})
    checks.append({'path':'frozen_parameter_hash','expected_before':summary.get('frozen_parameter_hash_before'),'observed_after':summary.get('frozen_parameter_hash_after'),'match':summary.get('frozen_parameter_hash_before')==summary.get('frozen_parameter_hash_after')})
    checks.append({'path':'checkpoint_restore','match':summary.get('checkpoint_restore_predictions_equal') is True and summary.get('checkpoint_restore_max_logit_abs_error')==0.0})
    return checks

def selection_ok(run, dev_map):
    summary, hist = readj(run/'summary.json'), readj(run/'history.json')
    # F1, accuracy, then earliest epoch.
    chosen = max(hist, key=lambda x: (x['dev']['binary_f1'], x['dev']['accuracy'], -x['epoch']))
    rows,ids,y,p,issues=load_predictions(run/'dev_predictions.jsonl',dev_map,dev_map.keys())
    dm=metric(y,p); target=summary.get('selected_internal_dev',{})
    restored=summary.get('restored_development_metrics',{})
    dev_metric_match=all(abs(dm[k]-target.get(k,float('nan')))<1e-12 and abs(dm[k]-restored.get(k,float('nan')))<1e-12 for k in ['accuracy','binary_f1'])
    return {'selected_epoch_reported':summary['selected_epoch'], 'selected_epoch_recomputed':chosen['epoch'],
            'matches':summary['selected_epoch']==chosen['epoch'] and [x['epoch'] for x in hist]==list(range(1,summary['epochs']+1)) and not issues and len(rows)==734 and dev_metric_match and all(abs(dm[k]-chosen['dev'][k])<1e-12 for k in ['accuracy','binary_f1']),
            'history_epochs':[x['epoch'] for x in hist], 'selected_internal_dev_recomputed':chosen['dev'], 'dev_prediction_metrics':dm,'dev_prediction_issues':issues}

def maybe_plots(records, out):
    try:
        import matplotlib.pyplot as plt
    except ImportError: return ['matplotlib unavailable']
    figs = []; complete = [r for r in records if r.get('valid') and r['kind']=='primary']
    if complete:
        by = defaultdict(list)
        for r in complete: by[(r['epochs'],r['method'])].append(r['metrics']['binary_f1'])
        labels, vals, errs = zip(*[(f'e{k[0]} {k[1]}', np.mean(v), np.std(v,ddof=1) if len(v)>1 else 0) for k,v in sorted(by.items())])
        fig, ax = plt.subplots(figsize=(8,4)); xx=np.arange(len(labels)); ax.bar(xx, vals, yerr=errs, capsize=3, color='#4269a5',alpha=.8)
        for j,key in enumerate(sorted(by)):
            points=by[key]; ax.scatter(np.full(len(points),j),points,color='#111111',s=18,zorder=3,label='individual seed' if j==0 else None)
        ax.set_xticks(xx,labels); ax.set_ylim(0,1); ax.set_ylabel('MRPC validation binary F1'); ax.tick_params(axis='x', rotation=25); ax.set_title('MRPC method and epoch budget: mean ± seed SD'); ax.legend(); fig.tight_layout(); p=out/'figure_mrpc_effects.png'; fig.savefig(p,dpi=180); fig.savefig(p.with_suffix('.svg')); plt.close(fig); figs.append(p.name)
    histories=[r for r in records if r.get('valid') and r['kind']=='primary' and (r['run']/'history.json').exists()]
    if histories:
        fig, ax=plt.subplots(figsize=(8,4)); grouped=defaultdict(list)
        for r in histories: grouped[(r['method'],r['epochs'])].append(readj(r['run']/'history.json'))
        for (m,e), hs in sorted(grouped.items()):
            epochs=sorted(set(x['epoch'] for h in hs for x in h)); means=[]; sds=[]
            for ep in epochs:
                vals=[next(x['training_loss'] for x in h if x['epoch']==ep) for h in hs if any(x['epoch']==ep for x in h)]
                means.append(np.mean(vals)); sds.append(np.std(vals,ddof=1) if len(vals)>1 else 0.)
                ax.scatter([ep]*len(vals),vals,s=12,alpha=.45)
            ax.errorbar(epochs,means,yerr=sds,marker='o',capsize=3,label=f'{m}, {e} epochs')
        ax.set(xlabel='Epoch',ylabel='Training loss',title='Mean ± seed SD; dots are individual seeds'); ax.legend(fontsize=7); fig.tight_layout(); p=out/'figure_loss_curves.png'; fig.savefig(p,dpi=180); fig.savefig(p.with_suffix('.svg')); plt.close(fig); figs.append(p.name)
    return figs

def paws_plot(paws, records, out):
    if paws.get('status') != 'COMPLETE': return []
    try:
        import matplotlib.pyplot as plt
    except ImportError: return ['matplotlib unavailable for PAWS figure']
    methods=['full','lora_r8']; x=np.arange(2); width=.34
    fig, axes=plt.subplots(1,2,figsize=(9.5,4.3))
    for index,(dataset,color,offset) in enumerate([('MRPC','#4269a5',-width/2),('PAWS','#d88c52',width/2)]):
        values=[]
        for method in methods:
            if dataset=='MRPC':
                values.append([r['metrics']['balanced_accuracy'] for r in records if r.get('valid') and r['kind']=='primary' and r['method']==method and r['epochs']==12])
            else:
                values.append([r['metrics']['balanced_accuracy'] for r in paws['records'] if r['passed'] and r['name'].startswith(method+'-')])
        axes[0].bar(x+offset,[np.mean(v) for v in values],width,yerr=[np.std(v,ddof=1) for v in values],capsize=3,color=color,label=dataset,alpha=.8)
        for j,v in enumerate(values):axes[0].scatter([j+offset]*len(v),v,color='black',s=12,zorder=3)
    neg=[[r['metrics']['negative_recall'] for r in paws['records'] if r['passed'] and r['name'].startswith(m+'-')] for m in methods]
    axes[1].bar(x,[np.mean(v) for v in neg],.55,yerr=[np.std(v,ddof=1) for v in neg],capsize=3,color='#d88c52',alpha=.8)
    for j,v in enumerate(neg):axes[1].scatter([j]*len(v),v,color='black',s=12,zorder=3)
    axes[0].axhline(.5,color='gray',linestyle='--',linewidth=1,label='Constant prediction balanced accuracy')
    for ax,title,ylabel in [(axes[0],'Same fitted models across datasets','Balanced accuracy'),(axes[1],'PAWS negative-class recall','Recall for non-paraphrase pairs')]:
        ax.set_xticks(x,methods);ax.set_ylim(0,1);ax.set_ylabel(ylabel);ax.set_title(title)
    axes[0].legend(fontsize=7);fig.suptitle('Mean ± sample SD over three seeds; dots are seeds',fontsize=10);fig.tight_layout()
    names=[]
    for ext in ['png','svg']:
        p=out/f'figure_mrpc_paws.{ext}';fig.savefig(p,dpi=180);names.append(p.name)
    plt.close(fig);return names

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--asset-dir',type=Path,required=True); ap.add_argument('--paws-dir',type=Path); ap.add_argument('--runtime-root',type=Path); ap.add_argument('--results-dir',type=Path,default=ROOT/'experiments/results'); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    out=args.out; out.mkdir(parents=True,exist_ok=True); protocol=readj(ROOT/'protocol.json'); assets=args.asset_dir
    train=pd.read_parquet(assets/'data/train-00000-of-00001.parquet'); val=pd.read_parquet(assets/'data/validation-00000-of-00001.parquet')
    train_map=dict(zip(train.idx,train.label)); val_map=dict(zip(val.idx,val.label)); records=[]; missing=[]; problems=[]
    for method,e,seed,rep,kind in expected(protocol):
        name=f'{method}-e{e}-seed{seed}'+('-repeat' if rep else ''); run=args.results_dir/name
        if not run.exists(): missing.append(name); continue
        required=['summary.json','history.json','predictions.jsonl','split_manifest.json','environment.json']
        absent=[x for x in required if not (run/x).exists()]
        r={'name':name,'method':method,'epochs':e,'seed':seed,'kind':kind,'run':run,'valid':not absent,'missing_files':absent}
        if absent: records.append(r); problems += [f'{name}:missing:{x}' for x in absent]; continue
        try:
            summary=readj(run/'summary.json'); split=readj(run/'split_manifest.json'); dev=split.get('splits',{}).get('dev',{}); dev_map=dict(zip(dev.get('ids',[]),dev.get('labels',[])))
            if len(dev_map)!=734 or not set(dev_map).issubset(train_map) or any(train_map.get(i)!=label for i,label in dev_map.items()): raise ValueError('invalid 734-row internal-development manifest')
            rows,ids,y,p,iss=load_predictions(run/'predictions.jsonl',val_map,val_map.keys())
            rec=metric(y,p); diffs={k:abs(rec[k]-summary['metrics'][k]) for k in ('accuracy','binary_f1')}
            strat=strata(val,ids,y,p); add_token_lengths(strat,val,ids,y,p,assets)
            r.update(summary=summary,metrics=rec,validation_check={'expected_rows':len(val),'rows':len(rows),'metric_abs_difference':diffs,'passed':len(rows)==len(val) and not iss and all(x<1e-12 for x in diffs.values())}, prediction_issues=iss, source_checks=verify_source(run,assets,args.runtime_root), selection_check=selection_ok(run,dev_map), strata=strat, _ids=ids.tolist(),_y=y.tolist(),_p=p.tolist())
            r['valid']=r['validation_check']['passed'] and r['selection_check']['matches'] and all(x.get('match',True) for x in r['source_checks'])
            if not r['valid']: problems.append(name+':validation_or_provenance_failure')
        except Exception as exc: r['valid']=False; r['analysis_error']=repr(exc); problems.append(name+':'+repr(exc))
        records.append(r)
    initial_heads={}
    for r in records:
        if r.get('valid'):
            key=str(r['seed']); initial_heads.setdefault(key,[]).append((r['name'],r['summary'].get('initial_classifier_sha256')))
    head_check={seed:{'values':vals,'match':len({v for _,v in vals})==1 and all(isinstance(v,str) and len(v)==64 for _,v in vals)} for seed,vals in initial_heads.items()}
    if any(not x['match'] for x in head_check.values()): problems.append('same_seed_initial_classifier_mismatch')
    # Training/dev/validation gap and group aggregate.
    for r in records:
        if r.get('valid'):
            s=r['summary']; r['gaps']={'training_minus_internal_dev_f1':s['selected_model_training_metrics']['binary_f1']-s['selected_internal_dev']['binary_f1'], 'internal_dev_minus_validation_f1':s['selected_internal_dev']['binary_f1']-r['metrics']['binary_f1']}
            diag=r['run']/'adapter_diagnostics.json'
            if diag.exists():
                ds=readj(diag); mods=ds if isinstance(ds,list) else ds.get('modules', [])
                rank=[]
                for x in mods:
                    sv=np.asarray(x.get('singular_values',[]),dtype=float); energy=np.cumsum(sv**2)/max(float(np.sum(sv**2)),1e-300); prob=(sv**2)/max(float(np.sum(sv**2)),1e-300)
                    rank.append({'module':x.get('module'),'numerical_rank':int(np.sum(sv>1e-8)),'energy_95_rank':int(np.searchsorted(energy,.95)+1),'energy_99_rank':int(np.searchsorted(energy,.99)+1),'entropy_effective_rank':float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0]))))})
                r['adapter_diagnostics']={'modules':len(mods),'mean_relative_update_norm':float(np.mean([x['relative_update_norm'] for x in mods])) if mods else None,'rank_diagnostics':rank}
    aggs={}
    for e in protocol['budget_epochs']:
        for m in protocol['primary_methods']:
            xs=[r for r in records if r.get('valid') and r['kind']=='primary' and r['epochs']==e and r['method']==m]
            if len(xs)==3:
                aggs[f'{m}-e{e}']={k:{'mean':float(np.mean([x['metrics'][k] for x in xs])),'sd':float(np.std([x['metrics'][k] for x in xs],ddof=1)),'per_seed':{str(x['seed']):x['metrics'][k] for x in xs}} for k in ['accuracy','binary_f1','balanced_accuracy','mcc','negative_recall','positive_recall']}
    pairs={}
    for e in protocol['budget_epochs']:
        for a,b in [('lora_r8','head'),('lora_r8','full'),('full','head')]:
            per={}
            for s in protocol['primary_seeds']:
                ra=next((r for r in records if r.get('valid') and (r['method'],r['epochs'],r['seed'],r['kind'])==(a,e,s,'primary')),None); rb=next((r for r in records if r.get('valid') and (r['method'],r['epochs'],r['seed'],r['kind'])==(b,e,s,'primary')),None)
                if ra and rb and ra['_ids']==rb['_ids']: per[str(s)]=bootstrap(np.array(ra['_y']),np.array(ra['_p']),np.array(rb['_y']),np.array(rb['_p']),protocol['error_analysis']['bootstrap_seed']+s,protocol['error_analysis']['bootstrap_samples'])
            pairs[f'{a}_minus_{b}-e{e}']=per
    for m in protocol['primary_methods']:
        per={}
        for s in protocol['primary_seeds']:
            a=next((r for r in records if r.get('valid') and (r['method'],r['epochs'],r['seed'],r['kind'])==(m,12,s,'primary')),None); b=next((r for r in records if r.get('valid') and (r['method'],r['epochs'],r['seed'],r['kind'])==(m,4,s,'primary')),None)
            if a and b and a['_ids']==b['_ids']: per[str(s)]=bootstrap(np.array(a['_y']),np.array(a['_p']),np.array(b['_y']),np.array(b['_p']),protocol['error_analysis']['bootstrap_seed']+10000+s,protocol['error_analysis']['bootstrap_samples'])
        pairs[f'{m}_e12_minus_e4']=per
    # old V3 values are a malformed-input diagnostic only.
    legacy=ROOT/'legacy_fault_diagnostic'/'results'
    legacy_compare=[]
    if legacy.exists():
        for r in records:
            if r.get('valid') and r['epochs']==4 and r['kind']=='primary':
                old=legacy/f"{r['method']}-seed{r['seed']}"/'summary.json'
                if old.exists(): legacy_compare.append({'run':r['name'],'legacy_path':str(old),'legacy_metrics':readj(old).get('metrics'), 'corrected_metrics':r['metrics'], 'scope':'faulty-tokenization diagnostic; not a normal fine-tuning comparison'})
    primary_repeat=next((r for r in records if r.get('valid') and r['name']=='lora_r8-e4-seed42'),None)
    repeat=next((r for r in records if r.get('valid') and r['name']=='lora_r8-e4-seed42-repeat'),None)
    repeat_check={'status':'NOT_AVAILABLE'}
    if primary_repeat and repeat:
        aa=[json.loads(x) for x in (primary_repeat['run']/'predictions.jsonl').read_text().splitlines() if x.strip()]
        bb=[json.loads(x) for x in (repeat['run']/'predictions.jsonl').read_text().splitlines() if x.strip()]
        fields={'same_row_order': [x['idx'] for x in aa]==[x['idx'] for x in bb], 'same_predictions': [x['prediction'] for x in aa]==[x['prediction'] for x in bb], 'same_logits': [x['logits'] for x in aa]==[x['logits'] for x in bb], 'same_checkpoint_sha256':primary_repeat['summary'].get('best_checkpoint_sha256')==repeat['summary'].get('best_checkpoint_sha256'), 'same_selected_epoch':primary_repeat['summary'].get('selected_epoch')==repeat['summary'].get('selected_epoch'), 'same_prediction_sha256':primary_repeat['summary'].get('prediction_sha256')==repeat['summary'].get('prediction_sha256')}
        repeat_check={'status':'PASS' if all(fields.values()) else 'FAIL',**fields,'primary_prediction_sha256':primary_repeat['summary'].get('prediction_sha256'),'repeat_prediction_sha256':repeat['summary'].get('prediction_sha256')}
    serial=[]
    for r in records:
        x={k:v for k,v in r.items() if k not in {'run','_ids','_y','_p'}}; serial.append(x)
    paws={'status':'NOT_REQUESTED'} if args.paws_dir is None else paws_audit(args.paws_dir,args.results_dir,protocol,args.runtime_root)
    status='COMPLETE' if not missing and not problems and repeat_check.get('status')=='PASS' and paws['status']=='COMPLETE' else 'INCOMPLETE'
    result={'analysis_status':status,'expected_runs':21,'found_runs':len(records),'missing_runs':missing,'problems':problems,'same_seed_initial_classifier':head_check,'records':serial,'aggregates_complete_three_seed_only':aggs,'paired_seed_bootstrap_conditional':pairs,'repeat_consistency':repeat_check,'v3_faulty_input_diagnostic_only':legacy_compare,'paws':paws,'limitations':['No missing value imputation','Bootstrap conditions on each fitted training seed','No claim of population significance or generalization','Correct-tokenizer lengths recomputed from pinned tokenizer, not word-count approximations']}
    result['figures']=maybe_plots(records,out)+paws_plot(paws,records,out); writej(out/'analysis.json',result)
    print(json.dumps({'analysis_status':status,'found_runs':len(records),'missing_runs':len(missing),'problems':len(problems)},ensure_ascii=False))

if __name__=='__main__': main()
