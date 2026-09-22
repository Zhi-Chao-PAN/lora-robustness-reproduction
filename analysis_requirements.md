# 独立分析与轻量审阅

本目录 README 包含建环境、固定资产下载和重训入口。公开仓库的轻量审阅运行 `verify_public_bundle.py`，只核对汇总证据、主结论、文件排除项和链接，不需要第三方 Python 包。严格研究分析必须先完整复跑，以生成固定模型资产、逐行预测与运行时权重。

以下从本目录运行，以 README 建立的 `.venv`、`assets`、新运行 `runtime_replay` 与 `experiments_replay` 为例：

```bash
.venv/bin/python analyze_results.py \
  --asset-dir assets \
  --runtime-root runtime_replay \
  --results-dir experiments_replay/results \
  --paws-dir paws-data \
  --out analysis_replay
.venv/bin/python make_report.py --analysis analysis_replay/analysis.json --out analysis_replay/研究分析报告.md
```

分析要求18个主配置、2个rank诊断、1次从头重复和6个PAWS压力结果齐全。每个训练目录保存734行dev与408行validation；每个压力目录8000行，source列为 `id`，不使用dataframe行号替代。`experiments_replay/stress/locked_model_selection.json` 必须对应MRPC内部dev已选checkpoint。缺失或不一致均标 `INCOMPLETE`，正式报告入口拒绝生成结论。

分析逐条重算标签/argmax/混淆矩阵与指标，检查冻结前源码、资产字节、协议、选优、恢复与所有原始权重。公开仓库不分发任何 checkpoint 或逐行预测；只运行轻量审阅时必须接受其较窄的核验范围，不能把摘要里的 SHA 说成自己已经读过权重字节。

条件bootstrap对每个固定种子的同一批验证句对采样2000次，三种子SD另列。错误分层按金标、词集合Jaccard和未截断token长度；空层标N/A。r2/r16是单种子诊断。九组被撤回的v3摘要和预测仅放在 `legacy_fault_diagnostic/` 作为错误输入条件记录。

原始运行位置不属于复现协议；复跑请始终显式传入自己的资产、运行和结果目录。最终分析状态与每次验证的源 SHA 另由交付清单冻结。
