# 变更日志

本文档记录项目的所有重要变更，包含功能说明、版本信息和更新时间。

---

## [v0.1.2] - 2026-05-29 13:45

### 变更内容
- 定位并修复 MAF @tool 在网关上 500 的根因（Anthropic 路径）：MAF 的 `AnthropicClient` 默认 beta flags（`mcp-client` / `code-execution`）会变成 `anthropic-beta` 请求头，网关在「带 tools」时 500。清空 `additional_beta_flags=[]` 即可，已对齐生产代码。
- `test_13`：AnthropicClient 加 `additional_beta_flags=[]` + `max_tokens`（@tool 现在应通过）；保留「默认 flags → 500」根因对照行；Responses 探针由 ❌ 改为 ℹ️（本就预期不可用）；新增绕开 MAF 的原生 openai SDK 两轮工具探针（turn-1 vs turn-2），用于判定 OpenAI 的 500 是请求带 tools 还是续轮所致。
- `test_12`：主 `AnthropicClient` 同样加 `additional_beta_flags=[]`，否则其工具检查会撞同一个 500。

### 涉及文件
- `tests/test_13_maf_compatibility.py` — beta flags + max_tokens、根因对照、Responses→INFO、OpenAI 原生两轮隔离探针。
- `tests/test_12_anthropic_tools.py` — 主 AnthropicClient 改用 `additional_beta_flags=[]`。

---

## [v0.1.1] - 2026-05-29 13:06

### 变更内容
- 默认运行收窄为「只跑 section 13」（MAF 兼容性矩阵）。`run_all_tests.py` 默认只执行 `test_13`，回答「MAF 在网关上能否工作、该选哪个 client」这一个问题。
- 深度 MAF 诊断（07、12）与全部非 MAF section（01-06、08-11）一并移到 `--full` 之后；模块清单重命名为 `DEFAULT_MODULES` / `EXTRA_MODULES`。
- 文档（README、CLAUDE.md）同步更新为「默认 = section 13」。

### 涉及文件
- `run_all_tests.py` — 默认模块列表改为仅 `test_13`；07/12 与非 MAF section 归入 `EXTRA_MODULES`（`--full` 时运行）；更新 docstring 与运行模式标签。
- `README.md` — 顶部说明与测试表标注改为「默认只跑 13」。
- `CLAUDE.md` — 运行命令与模块列表说明同步更新。

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
