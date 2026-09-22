"""Build a quantitative, bounded report only from verified experiment outputs."""
import argparse
import json
from pathlib import Path
import statistics


def mean(xs): return statistics.mean(xs)
def sd(xs): return statistics.stdev(xs) if len(xs)>1 else 0.0
def fmt(x): return f'{x:.4f}'
def cell(obj): return f"{obj['mean']:.4f} ± {obj['sd']:.4f}"
def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
    x=json.loads(a.analysis.read_text());complete=x['analysis_status']=='COMPLETE'
    if not complete and not a.allow_incomplete:raise SystemExit('Refusing a final report: independent analysis is incomplete.')
    records=[r for r in x['records'] if r.get('valid')];primary=[r for r in records if r['kind']=='primary'];ag=x['aggregates_complete_three_seed_only'];pairs=x['paired_seed_bootstrap_conditional']
    outcome=''
    if complete:
        stress={m:mean([r['metrics']['balanced_accuracy'] for r in x['paws']['records'] if r['name'].startswith(m+'-')]) for m in ['full','lora_r8']}
        outcome=(f"**主要发现：**12轮时 LoRA-r8 的 MRPC 三种子平均 F1 为 {ag['lora_r8-e12']['binary_f1']['mean']:.4f}，full 为 {ag['full-e12']['binary_f1']['mean']:.4f}；"
                 f"同六个模型在 PAWS 的平均 balanced accuracy 分别为 {stress['lora_r8']:.4f}、{stress['full']:.4f}，接近常量分类基线 .5。"
                 '在这组实验中，参数高效微调保留了接近全量微调的常规分数，但两者都没有在压力集上表现出可靠的词序区分能力；小幅分差不构成方法等效性证明。')
    lines=['# LoRA 的参数效率能否同时保住常规效果与词序鲁棒性？','', '**状态：独立分析完整。**' if complete else '**未完成进度报告，不可作正式结论。**','',outcome,'',
    '## 问题与输入纠错','',
    '本实验研究输入正确后，微调方式与训练预算如何影响 MRPC 句对判断，并检查这些已选模型在 PAWS 高词汇重叠句对上是否仍能区分语义差异。主比较是分类头训练、全量微调、query/value LoRA-r8；预算为 4 与 12 epoch，各做三个训练种子。','',
    'v3 采用的 legacy RobertaTokenizerFast 在固定环境中构造出零 BPE merges 的后端，4,076 条 MRPC 句对全部与原始 tokenizer.json 编码不符，绝大多数因字符级切分超过 128 token。v3 的正常微调性能解释已撤回；此错误来自本项目输入入口与核验遗漏，不归咎于上游未经验证的缺陷。','',
    'v4 使用 AutoTokenizer，并逐条与原始 tokenizers.Tokenizer 对照实际缓存的 input_ids、attention_mask 和 label。正确编码的 MRPC 样本没有超过 128 token；PAWS 8,000 行也逐条验证，其中 2 行超过 128，并按预设规则截断。PAWS 与 MRPC 无规范化无序句对的精确交集；这不等于语义或预训练数据完全独立。','',
    '## 方法、选择与原论文差异','',
    '固定 RoBERTa-base revision 与 MRPC revision；官方训练集分为 2,934 条梯度训练、734 条内部开发。每个 epoch 按内部开发 F1、accuracy、最早 epoch 依次选优；官方 408 条 validation 用于最终评价。该公开验证集此前已被查看，所有扩展按探索性研究解释。未下载 MRPC 官方 test。','',
    '仅训练分类头的基线先冻结编码器，缓存 eval-mode 编码器特征，再训练原 dense/tanh/dropout 头；全量微调更新全部参数；LoRA 更新 24 个 query/value 投影中的低秩矩阵及分类头。ΔW=(alpha/r)BA，每个投影新增 r(d_in+d_out) 参数，alpha=2r。固定方法学习率分别为 .001、2e-5、4e-4，有效 batch=32，长度128，BF16，AdamW。','',
    '4 与 12 epoch 使用各自完整的线性 warmup/decay 日程，因此预算对照不是在同一条轨迹上截取第4轮。r2/r16 各仅 seed42；它们提供机制诊断，不用于选择“最优秩”。原作者 MRPC 脚本使用 MNLI 适配后的初始化、30轮、长度512和多卡配置，且论文汇总口径不同，本实验不复制原论文表格。来源：[LoRA论文](https://arxiv.org/abs/2106.09685)、[固定作者脚本](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/examples/NLU/roberta_base_mrpc.sh)。','',
    '## MRPC 主结果','',
    '下表为三个训练种子的均值 ± 样本标准差；每项均在相同408条句对上评估。SD 表示有限训练种子差异，不是总体置信区间。','',
    '|方法/预算|F1|Accuracy|Balanced accuracy|负类召回|MCC|','|---|---:|---:|---:|---:|---:|']
    for e in [4,12]:
        for m in ['head','full','lora_r8']:
            key=f'{m}-e{e}'
            if key in ag:
                v=ag[key];lines.append(f"|{key}|{cell(v['binary_f1'])}|{cell(v['accuracy'])}|{cell(v['balanced_accuracy'])}|{cell(v['negative_recall'])}|{cell(v['mcc'])}|")
    if primary:
        cm=primary[0]['metrics']['confusion_matrix'];neg=cm[0][0]+cm[0][1];pos=cm[1][0]+cm[1][1];n=pos+neg
        lines+=['',f'验证集正类 {pos}/{n}。恒预测正类的 F1 为 {2*pos/(n+pos):.6f}、accuracy 为 {pos/n:.6f}、balanced accuracy 为 .5。因此单看 F1 不足以评价少数负类。']
    if all(k in ag for k in ['lora_r8-e12','full-e12','lora_r8-e4','full-e4']):
        for e in [4,12]:
            d=ag[f'lora_r8-e{e}']['binary_f1']['mean']-ag[f'full-e{e}']['binary_f1']['mean']
            lines+=['',f'{e}轮时 LoRA−全量微调的三种子平均 F1 差为 {d:+.6f}。这是本固定协议下的描述差异，不能由小差值直接宣称两者等效或 LoRA 普遍更优。']
    lines+=['','![主结果与各训练种子](figure_mrpc_effects.png)','','## 预算、拟合与配对不确定性','','|方法|12−4轮 平均F1变化|三个seed的配对差值|12轮所选epoch|','|---|---:|---|---|']
    for m in ['head','full','lora_r8']:
        if f'{m}-e4' in ag and f'{m}-e12' in ag:
            v=pairs[f'{m}_e12_minus_e4'];diff=ag[f'{m}-e12']['binary_f1']['mean']-ag[f'{m}-e4']['binary_f1']['mean']
            ep=[r['summary']['selected_epoch'] for r in primary if r['method']==m and r['epochs']==12]
            lines.append(f"|{m}|{diff:+.6f}|"+', '.join(f"{s}: {v[s]['binary_f1']['observed_difference']:+.4f}" for s in sorted(v,key=int))+f"|{ep}|")
    lines+=['','对每个固定种子的两个模型，在相同验证样本上配对 bootstrap 2,000 次，保持种子预设。以下给出 LoRA−全量的 F1 差和95%百分位区间；完整六指标及预算配对在 analysis.json。区间条件于已拟合模型和当前样本集合，不包含重新选参、训练种子总体或领域迁移不确定性。多项探索性比较未作多重检验校正。','','|预算|训练seed|观察F1差|95%配对区间|','|---|---:|---:|---|']
    for e in [4,12]:
        for seed,v in sorted(pairs.get(f'lora_r8_minus_full-e{e}',{}).items(),key=lambda item:int(item[0])):
            z=v['binary_f1'];lo,hi=z['ci95_percentile'];lines.append(f"|{e}|{seed}|{z['observed_difference']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|")
    lines+=['','![训练损失与种子差异](figure_loss_curves.png)','','|12轮方法|选优模型训练F1均值|内部dev F1均值|官方validation F1均值|','|---|---:|---:|---:|']
    for m in ['head','full','lora_r8']:
        rs=[r for r in primary if r['method']==m and r['epochs']==12]
        if len(rs)==3:lines.append(f"|{m}|{mean([r['summary']['selected_model_training_metrics']['binary_f1'] for r in rs]):.4f}|{mean([r['summary']['selected_internal_dev']['binary_f1'] for r in rs]):.4f}|{mean([r['metrics']['binary_f1'] for r in rs]):.4f}|")
    lines+=['','内部 dev 和官方 validation 来自不同划分，样本难度未必相同；dev 用于选 epoch，本表仅诊断拟合差距，不把 validation 比 dev 高误读为不存在过拟合。','','## PAWS：常规分数之外的词序压力','','[PAWS](https://arxiv.org/abs/1904.01130) 构造高词汇重叠、词序和句意关系困难的句对。固定六个12轮 full/LoRA 模型的 MRPC 内部 dev 选优 checkpoint 后，才进行 PAWS-Wiki labeled_final 公开 test 的零样本评价；没有 PAWS 训练、选 epoch 或阈值调优。它是特定压力集，不能代表所有真实场景。','']
    paws=x['paws'];lines.append(f"逐条压力预测审计状态：`{paws['status']}`。")
    if paws['status']=='COMPLETE':
        lines+=['','|方法|Accuracy|F1|Balanced accuracy|负类召回|正类召回|MCC|','|---|---:|---:|---:|---:|---:|---:|']
        for m in ['full','lora_r8']:
            rs=[r['metrics'] for r in paws['records'] if r['name'].startswith(m+'-')]
            vals=[f"{mean([r[k] for r in rs]):.4f} ± {sd([r[k] for r in rs]):.4f}" for k in ['accuracy','binary_f1','balanced_accuracy','negative_recall','positive_recall','mcc']]
            lines.append('|'+m+'|'+'|'.join(vals)+'|')
        lines+=['','该集合有4,464条负类、3,536条正类。恒预测负类 accuracy=.558，恒预测正类 F1约.613；两者 balanced accuracy均=.5，MCC在本实现零分母约定下记0。压力结果应优先结合 balanced accuracy 与负类召回，不能只比较 F1。',
        '', '六个模型的正类召回都接近1，负类召回都低于.006，几乎把所有句对判为同义；约.613的F1因此不表示有效识别了非同义句对。这一负面结果限制了MRPC成绩的外推。它与词汇重叠、训练标签比例或数据迁移有关的解释仍是假设，当前实验没有分离各因素的因果作用。表内四位小数可能把很小的SD显示为.0000，完整精度保留在JSON。','', '![同一模型的常规与压力表现](figure_mrpc_paws.png)']
    lines+=['','## 错误分层与低秩机制','','词汇重叠由小写词集合 Jaccard 衡量，仅作描述性分层；不是“理解语义”的直接测量。固定顺序选取的事后失败案例包含第三方数据集原句，未纳入公开仓库；完整复跑可由 error_cases.py 重新生成。以下显示12轮每方法在各层的三种子平均错误率，n为验证句对数，三个seed不使样本量翻三倍。','','|分层|n|head错误率|full错误率|LoRA错误率|','|---|---:|---:|---:|---:|']
    for band in ['gold_0','gold_1','jaccard_[0,.5)','jaccard_[.5,.8)','jaccard_[.8,1]','length_<=64','length_65_128','length_>128']:
        vals=[];n=0
        for m in ['head','full','lora_r8']:
            z=[r['strata'][band] for r in primary if r['method']==m and r['epochs']==12]
            if z:n=z[0]['n']
            vals.append(f"{mean([q['error_rate'] for q in z]):.4f}" if z and z[0]['n'] else 'N/A')
        lines.append(f'|{band}|{n}|'+'|'.join(vals)+'|')
    lines+=['','|12轮rank诊断（seed42）|F1|可训练参数含头|24个投影平均相对更新范数|95%能量秩均值|','|---|---:|---:|---:|---:|']
    for m in ['lora_r2','lora_r8','lora_r16']:
        rs=[r for r in records if r['method']==m and r['epochs']==12 and r['seed']==42]
        if rs:
            r=rs[0];d=r['adapter_diagnostics'];rk=d['rank_diagnostics'];lines.append(f"|{m}|{r['metrics']['binary_f1']:.4f}|{r['summary']['trainable_parameters']:,}|{d['mean_relative_update_norm']:.6f}|{mean([q['energy_95_rank'] for q in rk]):.2f}|")
    lines+=['','相对范数为 ||ΔW||F/||W||F；能量秩按奇异值平方累计计算。数值秩的阈值固定1e-8，另存99%能量秩与熵有效秩。低秩结构由参数化强制形成，不能把“秩低”本身当成任务具有内在低维结构的实验证明；各层也不是独立训练重复。','','## 参数、成本与重复','','|12轮方法|可训练参数|峰值CUDA分配GiB均值|端到端秒数均值|','|---|---:|---:|---:|']
    for m in ['head','full','lora_r8']:
        rs=[r['summary'] for r in primary if r['method']==m and r['epochs']==12]
        if len(rs)==3:lines.append(f"|{m}|{rs[0]['trainable_parameters']:,}|{mean([r['peak_torch_cuda_allocated_bytes']/2**30 for r in rs]):.3f}|{mean([r['wall_seconds'] for r in rs]):.1f}|")
    if any(r['method']=='full' for r in primary) and any(r['method']=='lora_r8' for r in primary):
        full=next(r['summary']['trainable_parameters'] for r in primary if r['method']=='full');low=next(r['summary']['trainable_parameters'] for r in primary if r['method']=='lora_r8');lines+=['',f'LoRA-r8 含头可训练参数较 full 减少 {100*(1-low/full):.3f}%。这不是总模型存储量或运行时间按相同比例减少；冻结编码器仍参与前向和反向传播所需的计算。时间包含方法相关的特征缓存、评价、权重保存与恢复，显存为 PyTorch allocator峰值，不是整机峰值，均不能当严格硬件基准。']
    lines+=['',f"从头重复 `lora_r8-e4-seed42` 的状态为 **{x['repeat_consistency']['status']}**：检查逐条预测、logits、checkpoint SHA、预测 SHA 和所选 epoch。仅证明这一配置在当前宿主环境的重复一致性，未验证不同GPU或软件版本。",'',
    '## 可复核交付与范围','',
    f"原始研究归档发现 {x['found_runs']}/21 个运行，缺失 {len(x['missing_runs'])}，独立校验问题 {len(x['problems'])}。训练前冻结协议、源码、依赖和资产哈希；原始归档保存每次734条dev与408条validation预测、训练损失、选优记录、恢复一致性、小权重与层级更新谱，并在运行环境核对 full 权重 SHA。公开仓库仅保留汇总记录和哈希，不分发预测或权重。",'',
    '公开轻量复核运行 `verify_public_bundle.py`：核对汇总证据与主结论，不加载模型。严格分析须先完整复跑，再运行 `analyze_results.py` 并提供固定资产、逐行预测与运行时权重；完整重训入口是 `run_suite.py`。报告从 analysis.json 自动构建，没有填补缺失结果。','',
    '局限集中在：一个骨干和两个公开英文句对数据集，三个训练种子，单一内部划分，各方法固定但未等预算充分调优的学习率，以及单个 PAWS 压力集。本轮由 AI 辅助实现、执行与分析；该交付不证明任何个人在无辅助条件下的独立掌握程度。']
    a.out.write_text('\n'.join(lines)+'\n');print(a.out)

if __name__=='__main__':main()
