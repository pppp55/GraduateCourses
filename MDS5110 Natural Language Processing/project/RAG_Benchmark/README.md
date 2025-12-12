# LegalRAG-Bench

面向法律检索场景的离线 Benchmark 管线，包含数据加载、检索器封装、指标计算（NDCG / Recall / MRR）以及 bad-case 分析工具。默认使用 `data/query_law_ids_validated.json` 中人工验证过的 Query-Law 对作为评测集合（附带 `source`、`detailed_source`、`law_contents` 等上下文），`data/法律法规.jsonl` 作为法条语料。

## 目录结构
```
LegalRAG-Bench/
├── configs/                # 跑 benchmark 的配置（default/remote_bm25/remote_hybrid 等）
├── data/                   # queries + 法条语料副本
├── outputs/                # 运行后生成日志、报告、bad cases
├── src/
│   ├── retrievers/         # `lexical`（离线 TF-IDF）与 `remote`（HTTP）
│   ├── metrics/            # NDCG / Recall / MRR 计算
│   └── utils/              # 数据加载、报告导出工具
├── analyze_results.py      # 对 outputs 进行二次分析
└── run_benchmark.py        # 主入口
```

## 配置检索方式
`configs/*.json` 控制数据路径、retriever 类型以及输出文件。所有配置现在支持：

- `run_name`：用于在 `outputs/<run_name>/` 下自动建立隔离的日志、报告、bad cases 文件夹，避免不同实验互相覆盖。
- `metadata`：写入 `metrics.json`，方便记录“这次 run 用的是什么检索模式/模型/超参”。

`configs/default.json` 示例：
```json
{
  "run_name": "remote_default",
  "data": {
    "queries_path": "data/query_law_ids_validated.json",
    "law_corpus_path": "data/法律法规.jsonl"
  },
  "retriever": {
    "type": "remote",
    "top_k": 10,
    "endpoint": "http://127.0.0.1:8006/retrieve",
    "timeout": 10.0
  }
}
```
- **本地 TF-IDF（离线）**：将 `"type"` 改成 `"lexical"`，其余参数可忽略；无需外部服务，直接在内存中计算相似度。
- **HTTP 远程检索**：保留 `"type": "remote"`，把 `endpoint` 改为你实际运行的 RAG 服务（与 `rag_request_for_bench.py` 同协议：POST `{"queries": [query], "topk": K, "return_scores": true}`，返回 `document.id` & `score`）。如需要代理、自定义超时，可在 `retriever` 节点增设 `"proxies"`、`"timeout"` 字段。

> 若你已有自己的配置文件，可在运行脚本时通过 `--config configs/xxx.json` 指定。

## 运行 Benchmark
1. **确保数据就绪**：`data/query_law_ids_validated.json` 和 `data/法律法规.jsonl` 位于 `LegalRAG-Bench/data/`。如需更新，可直接覆盖该目录中的文件。
2. **启动检索服务（仅 Remote 模式）**：保证 `endpoint` 对应的本地/远程服务已在监听，并能返回预期结果。
3. **执行主脚本**：
   ```bash
   python run_benchmark.py \
     --config configs/default.json \
     --top_k 10 \
     --max_queries 10  # 可选：限制样本数，调试更快
   ```
   - `--top_k` 与配置文件中的 `retriever.top_k` 二选一；命令行存在时会覆盖配置。
   - `--max_queries` 用于抽样调试，大规模跑完后删除该参数即可。

运行结束后会看到日志输出，并在 `outputs/<run_name>/` 下生成：
- `logs/benchmark.log`：完整日志。
- `reports/predictions.json`：每个 query 的预测列表、得分、snippet、`law_texts`（GT 法条原文）、`bench_source`（题目来源/构造方式）以及单条指标，方便核验。
- `reports/metrics.(json|csv)`：整体平均指标（NDCG / Recall / MRR / hit-rate）、运行耗时、以及 `metadata` 中记录的检索模式与配置。
- `reports/per_source_metrics.csv`：按照 `bench_source` 汇总的各任务得分，便于分析系统在不同题源上的强弱。
- `bad_cases/diff_cases.json`：未命中的案例，携带 Query、GT 法条（含原文/来源）与 top-k 错误候选。

## 查看 & 分析结果
使用自带脚本快速查看：
```bash
python LegalRAG-Bench/analyze_results.py \
  --predictions LegalRAG-Bench/outputs/remote_hybrid/reports/predictions.json \
  --metrics LegalRAG-Bench/outputs/remote_hybrid/reports/metrics.json \
  --diff LegalRAG-Bench/outputs/remote_hybrid/bad_cases/diff_cases.json \
  --limit 5
```
- 会在控制台打印总体指标、运行耗时、各 `bench_source` 子任务得分以及前若干条失败案例（Query + Ground Truth + Wrong Retrievals + Bench Source），方便快速人工复核。
- 如果 `diff_cases.json` 不存在，脚本会现场根据 `predictions.json` 重新构建。

## 常见问题
1. **远程检索接口报错/超时**：
   - 检查 `endpoint` 是否可访问；可用 `curl` 或 `rag_request_for_bench.py` 先验证。
   - 如果服务返回的 JSON 结构不同，需调整 `src/retrievers/remote.py` 中 `_call_service` 的解析逻辑。
2. **没有命中 GT 导致分数为 0**：
   - 查看 `outputs/bad_cases/diff_cases.json` 或运行 `analyze_results.py`，确认 query 与 GT 对是否正确，或检索服务是否真正包含这些法条。
3. **想自定义新的检索器**：
   - 在 `src/retrievers/` 新增实现并在 `__init__.py` 的 `build_retriever` 注册即可；`run_benchmark.py` 会统一消费 `search(query, top_k)` 的输出。

## 与 `rag_request_for_bench.py` 的关系
- `rag_request_for_bench.py` 仍可单独使用（例如只想快速把检索结果打成 JSON）。
- `LegalRAG-Bench` 的 `remote` 检索器沿用了同样的 HTTP 协议，所以你可以直接用相同的服务地址，无需再次实现。

## 多检索模式配置示例

当使用 `retrieval_server_text2vec.py` 暴露的 HTTP 服务时，需要根据不同的 `retriever_name` 重新启动服务。下面给出常用的启动命令（示例与 `retrieval_launch_law_text2vec.sh` 对齐）：

```bash
export HTTP_PROXY=http://nas.betterspace.top:10809
export HTTPS_PROXY=http://nas.betterspace.top:10809
export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE='/caizhenyang/panghuaiwen/legal_LLM/model'
conda activate retriever

# 以 hybrid 为例
bash retrieval_launch_law_text2vec.sh \
  --gpu_ids 0 1 2 3 \
  --gpu_memory_limit_per_gpu 24 \
  --port 8006 \
  --corpus_path "/caizhenyang/panghuaiwen/legal_LLM/dataset/dataset/law/法律法规2.0.jsonl" \
  --retriever_name hybrid \
  --dictionary_path "/caizhenyang/panghuaiwen/legal_LLM/dataset/dataset/dictionary/THUOCL_law.txt" \
  --search_depth 5 \
  --bm25_weight 15
```

对应的 benchmark 配置：

| 配置文件 | run_name | metadata.retrieval_profile | 说明 |
|----------|----------|-----------------------------|------|
| `configs/remote_hybrid.json`   | `remote_hybrid`   | `hybrid`   | 使用上面命令启动的 hybrid 检索服务 |
| `configs/remote_bm25.json`     | `remote_bm25`     | `bm25`     | 启动脚本改为 `--retriever_name bm25` |
| `configs/remote_text2vec.json` | `remote_text2vec` | `text2vec` | 启动脚本改为 `--retriever_name text2vec` |

每次切换检索模式：
1. 重新启动 RAG 服务，确保 `--retriever_name` 与上表一致。
2. 运行对应的 `run_benchmark.py --config configs/remote_xxx.json`。
3. 各自的结果会落到 `outputs/remote_xxx/`，互不覆盖，方便比较。

按照以上流程修改配置 → 启动检索服务（如需）→ 执行 `run_benchmark.py` → 使用 `analyze_results.py` 查看结果，即可跑完完整的 RAG benchmark。祝评测顺利！
