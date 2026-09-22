"""Deterministic, post-hoc examples; never tune a model on these examples."""
import json
from pathlib import Path
import re
import pandas as pd

ROOT=Path(__file__).resolve().parent

def load(path):return {r['idx']:r for r in (json.loads(line) for line in path.read_text().splitlines())}
def overlap(a,b):
    x=set(re.findall(r'\w+',a.lower()));y=set(re.findall(r'\w+',b.lower()))
    return len(x&y)/len(x|y) if x|y else 1.0

def main():
    if json.loads((ROOT/'analysis/analysis.json').read_text())['analysis_status']!='COMPLETE':raise ValueError('Complete audit required')
    cases=[];counts={}
    for dataset,data,predroot in [('MRPC',ROOT/'input-assets/data/validation-00000-of-00001.parquet',ROOT/'experiments/results'),('PAWS',ROOT/'paws-data/test.parquet',ROOT/'experiments/stress')]:
        f=pd.read_parquet(data);identifier='idx' if dataset=='MRPC' else 'id';f=f.sort_values(identifier).set_index(identifier)
        a=load(predroot/'full-e12-seed42/predictions.jsonl');b=load(predroot/'lora_r8-e12-seed42/predictions.jsonl')
        buckets={'both_false_positive':[],'both_false_negative':[],'model_disagreement':[]}
        for idx,row in f.iterrows():
            idx=int(idx);gold=int(row.label);pa=a[idx]['prediction'];pb=b[idx]['prediction']
            if pa==pb==1 and gold==0:category='both_false_positive'
            elif pa==pb==0 and gold==1:category='both_false_negative'
            elif pa!=pb:category='model_disagreement'
            else:continue
            buckets[category].append({'dataset':dataset,'category':category,'source_id':idx,'sentence1':row.sentence1,'sentence2':row.sentence2,'label':gold,'full_prediction':pa,'lora_prediction':pb,'lexical_jaccard':overlap(row.sentence1,row.sentence2)})
        counts[dataset]={key:len(value) for key,value in buckets.items()}
        for category,items in buckets.items():
            if items:cases.append(items[0])
    payload={'scope':'Post-hoc qualitative examples, seed42 only; first source ID in each error category, no manual selection or tuning.','category_counts':counts,'cases':cases}
    (ROOT/'error_cases.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=['# 失败案例：固定顺序的事后诊断','','只比较12轮 seed42 的 full 与 LoRA-r8。每个数据集、每个错误类别取最小 source ID，不按故事好坏手挑。类别样本数是这两个固定模型在当前数据集的描述，示例不代表错误分布全部特征，也不用于再调模型。标签0=非同义，1=同义。','','完整句子见同目录 `error_cases.json` 与固定数据源；下表给编号和统计，便于定位。','','|数据集/类别|source ID|金标|full/LoRA预测|词集合Jaccard|该类句对数|','|---|---:|---:|---|---:|---:|']
    for c in cases:
        lines.append(f"|{c['dataset']} / {c['category']}|{c['source_id']}|{c['label']}|{c['full_prediction']} / {c['lora_prediction']}|{c['lexical_jaccard']:.3f}|{counts[c['dataset']][c['category']]}|")
    notes={
        ('MRPC',374):'两句保留相同人物和核心引语，但时间与讲话背景的表述不同；金标为0。单凭引语高度重叠不能覆盖数据集的整句判断标准，时间是否同指还需要新闻上下文。',
        ('MRPC',127):'关于法官要求为同一人提供监管住房和治疗的核心事件一致，第一句多出无法安置的背景；金标为1，两模型都拒绝了这一句对。',
        ('MRPC',143):'一个句子描述指数上涨，另一个描述下跌，点数与指数终值也不同。full正确判为0，LoRA判为1；这是可直接核对方向和数字差异的个案。',
        ('PAWS',1):'表示角度与极坐标的词被换到不同名词关系中，词集合Jaccard仍高达.852。金标为0，两模型都判为1；它展示词集合重叠无法保存词与对象的对应关系，但不证明模型内部只使用词袋。',
        ('PAWS',772):'第二句完整保留迁居和执业信息，并新增担任地方职务的片段；金标仍为1，两模型都判为0。这提示需要理解该数据集的同义标注口径，不能将标签直接等同于严格逻辑双向蕴含。',
        ('PAWS',89):'两句涉及zeta函数与beta，第一句语序较不流畅且带有额外条件词；金标为0，LoRA正确而full错误。流畅性和信息差异同时存在，此单例不能隔离词序的独立作用。',
    }
    lines+=['','## 逐例阅读记录','', '以下解释在结果冻结后补充，属于人工可核对的文本观察，不是模型决策机制归因。','']
    for c in cases:
        key=(c['dataset'],c['source_id'])
        if key in notes:lines.append(f"- **{c['dataset']} / {c['source_id']}：**{notes[key]}")
    lines+=['','如何审阅：先遮住预测阅读原句，再核对金标；记录实体、否定、数量与词序差异是否改变关系。模型共同出错不证明内部使用了某一种捷径，较高Jaccard也不构成因果解释。若不同意金标，应记录争议并另行设计审查，不能为当前成绩直接改标签。']
    (ROOT/'error_cases.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'case_count':len(cases),'category_counts':counts},ensure_ascii=False))

if __name__=='__main__':main()
