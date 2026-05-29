# 变更日志

本文档记录项目的所有重要变更，包含功能说明、版本信息和更新时间。

---

## [初始版本] - 2026-05-29 11:11

### 项目概述
ZF AI Gateway 兼容性与性能测试套件：针对公司内部 AI Assistant Suite 网关（`ai-assistant-suite-staging.azurewebsites.net`）的一次性诊断工具。在能访问网关的机器（公司服务器 / VPN）上运行，生成单一的 `results/TEST_REPORT.md` 供分析。当前重点已转向 **Microsoft Agent Framework（MAF）对网关的适用性测试**。

### 初始功能
- **MAF 兼容性矩阵（默认运行）**：`test_13_maf_compatibility` 并排测试三种 MAF client —— `OpenAIChatClient`（Responses `/responses`，预期 404）、`OpenAIChatCompletionClient`（Chat Completions，预期可用）、`AnthropicClient`（`/v1/messages`，预期可用），每种 client 跑 [连接 / 流式 / @tool / 双 agent workflow] 矩阵。
- **MAF 深度诊断**：`test_07`（MAF + OpenAIChatCompletionClient）与 `test_12`（MAF + AnthropicClient）保留为深度排查模块（原始 round-trip、beta-flag 开关等），用于定位 @tool 500 的真正来源。
- **默认 MAF-only，`--full` 跑全部**：`run_all_tests.py` 默认只跑 MAF 相关模块；非 MAF 的 01-06、08-11 保留在磁盘上，仅在 `--full` 时运行。
- **报错日志可调试**：`reporter.py` 的 `capture_exception()` 现在保存完整 traceback（含 SDK 嵌入的 HTTP 响应体），在报告中以可折叠的「🪵 Failure logs」块呈现。
- 非 MAF 探针（保留）：OpenAI SDK 基础连通性、参数支持、流式、多模态、函数调用、Anthropic SDK、并发压测、上下文窗口上限、reasoning/temperature 有效性。

### 技术栈
- Python 3.10+
- `agent-framework` 1.0（MAF），`agent-framework-anthropic`（`--pre`）
- `openai`、`anthropic` 官方 SDK
- `httpx`、`pydantic`、`tiktoken`、`python-dotenv`
- 测试资产生成：`Pillow`、`reportlab`、`pymupdf`、`PyPDF2`
