# LTX-Desktop 面试问答

## Q1：为什么 LTX Desktop 要采用 `React + Electron + FastAPI` 三层结构，而不是只用 Electron main process 直接做推理？

**参考答案：**

- Electron main process 更适合做桌面生命周期、窗口管理、IPC、文件系统和系统集成，不适合直接承接复杂的 Python/ML 推理逻辑。
- Python 生态本身更适合模型加载、diffusers、PyTorch、huggingface 等依赖管理。
- 引入本地 FastAPI 之后，前端拿到的是统一的 HTTP API，不需要知道底层到底是本地 GPU 还是云端 API。
- 这样也让测试、状态机、运行策略、模型下载、日志等能力更容易集中管理。

一句话：Electron 负责桌面能力，FastAPI 负责 AI 运行时，React 负责产品交互，这样边界最清楚。

---

## Q2：既然 backend 只监听 `127.0.0.1`，为什么还要加 auth token？

**参考答案：**

- `localhost` 不等于天然安全，尤其是桌面软件运行时可能同时存在多个本地进程或恶意脚本。
- LTX Desktop 在 Electron 启动 backend 时会生成会话级 `authToken` 和 `adminToken`，并通过环境变量传给 Python backend。
- 前端发 HTTP 请求时由 `backendFetch` 自动注入 `Bearer token`。
- 特权操作（例如修改 `models_dir`）还需要额外的 `X-Admin-Token`。

这体现的是“本地服务也要有零信任边界”的思路。

---

## Q3：这个项目是如何决定走本地推理还是 API-only 模式的？

**参考答案：**

- 决策入口在 `backend/runtime_config/runtime_policy.py`。
- `Darwin` 一律强制 API-only。
- `Windows/Linux` 如果没有 CUDA，或者 VRAM 未知，或者 `VRAM < 31GB`，都强制 API-only。
- 如果系统允许本地推理，用户还可以通过设置决定是否偏好 LTX API 视频生成。
- 真正执行时，`video_generation_handler.generate()` 会调用 `should_video_generate_with_ltx_api(...)` 决定最终路径。

这说明项目把“系统约束”和“用户偏好”分成了两层。

---

## Q4：为什么前端不直接 `fetch('http://localhost:8000')`，而是先通过 Electron 拿 backend 地址和 token？

**参考答案：**

- backend 端口未必固定，Electron 会在 Python 输出 ready message 后解析真实地址。
- backend token 是 Electron 会话生成的，前端不应该自己持有或硬编码。
- 把 backend 访问统一收口到 `frontend/lib/backend.ts`，后续要改重试、日志、SSE、错误处理都更容易。
- 这样还能避免组件层到处散落直连 backend 的逻辑。

本质上，这是把“网络边界”和“桌面会话边界”统一管理。

---

## Q5：`_routes -> AppHandler -> handlers -> services -> state` 这套后端分层有什么价值？

**参考答案：**

- `_routes` 保持很薄，只处理 HTTP 输入输出。
- `AppHandler` 是组合根，负责把共享状态、锁、配置和各类 handler 装配起来。
- `handlers` 承担业务逻辑和状态迁移。
- `services` 是 GPU、HTTP、文件 IO 等重副作用边界。
- `state` 负责集中表达 runtime state machine。

这种结构特别适合：

- 有共享状态
- 有后台任务
- 有重 IO / 重 GPU 逻辑
- 需要做 integration-style tests

相比把逻辑写在 endpoint 里，这种方式更清晰，也更容易替换 fake services 做测试。

---

## Q6：LTX Desktop 在“首启体验”和“更新体验”上做了哪些工程化设计？

**参考答案：**

- Windows/Linux 可以在首启下载 Python embed，减小安装包体积。
- `python-setup.ts` 通过 `deps-hash.txt` 判断当前 Python 环境是否匹配。
- 更新时支持把下一版本 Python 环境先下载到 `python-next/`，下次启动直接 promote。
- 模型与 Python 环境是分开的：Python 负责运行时，模型权重由 backend 在业务层单独下载。

这说明团队把“安装包体积、启动速度、升级可靠性”都考虑进去了。

---

## Q7：这个项目目前最大的可维护性风险在哪里？

**参考答案：**

- 前端大组件比较明显，尤其是 `GenSpace.tsx` 和 `VideoEditor.tsx`。
- 前端缺少与 backend 同等级的测试护栏。
- 当前进度同步主要依赖 polling，多个域的状态更新机制不统一。
- 项目数据保存在 `localStorage`，当项目和素材规模变大时会成为扩展瓶颈。

所以如果让我接手维护，我会优先做：

1. 大组件拆分
2. 前端 integration test
3. 统一事件流
4. 升级项目持久化层

---

## Q8：如果你来优化这个项目的生成状态同步机制，你会怎么做？

**参考答案：**

- 现状是前端轮询 `/api/generation/progress`，简单可靠，但扩展性一般。
- 我会优先考虑把 generation/download/export/backend-health 抽象成统一事件模型。
- 技术实现上可以选 `SSE` 或 `WebSocket`：
  - `SSE` 更简单，适合单向进度推送。
  - `WebSocket` 更灵活，适合未来增加控制命令或双向交互。
- 前端可以统一用一个 event store，避免每个 hook 自己写一套 polling 逻辑。

预期收益：

- UI 进度更平滑
- 请求数量更少
- 状态更新更一致
- 更容易支持多任务并发

---

## Q9：这个项目最值得你复用到自己产品里的设计是什么？

**参考答案：**

我会优先复用三点：

1. `Electron 负责会话、权限与本地编排，Python 负责 AI runtime`
2. `Local / API` 双模式统一通过同一 backend surface 暴露给前端
3. 本地服务依然做 token 鉴权、CSP 与路径校验，不因为是桌面软件就放松安全要求

这三点一起，基本就构成了一个 AI Desktop App 的工程骨架。
