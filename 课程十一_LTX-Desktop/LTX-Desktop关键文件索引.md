# LTX-Desktop 关键文件索引

## 1. 总览文档

| 文件 | 作用 | 为什么值得先读 |
|---|---|---|
| `README.md` | 项目定位、功能、运行模式、系统要求 | 先建立产品视角 |
| `backend/architecture.md` | 后端分层与并发/测试约束 | 理解代码组织的核心依据 |
| `docs/INSTALLER.md` | 打包、安装产物、Python bundling | 理解“桌面产品化”部分 |
| `docs/TELEMETRY.md` | 遥测边界与隐私说明 | 理解数据治理与隐私表达 |
| `docs/CONTRIBUTING.md` | 当前可接受改动范围 | 看出项目仍在前端重构期 |

## 2. Frontend 必读文件

| 文件 | 职责 | 阅读重点 |
|---|---|---|
| `frontend/App.tsx` | 应用总入口 | 首启、Python ready、backend start、全局 gating |
| `frontend/lib/backend.ts` | backend 统一访问层 | URL 与 token 是如何注入的 |
| `frontend/hooks/use-generation.ts` | 视频/图片生成 hook | 请求体、进度轮询、取消、结果回填 |
| `frontend/contexts/AppSettingsContext.tsx` | 设置与 runtime policy 对接 | 前端如何感知 API-only 模式 |
| `frontend/contexts/ProjectContext.tsx` | 项目/资产/时间线状态 | `localStorage` 持久化与项目数据模型 |
| `frontend/views/GenSpace.tsx` | 生成工作台 | Text/Image/Audio/Retake/IC-LoRA 多模式入口 |
| `frontend/views/VideoEditor.tsx` | 视频编辑工作台 | 时间线、导出、项目编辑流 |

## 3. Electron 必读文件

| 文件 | 职责 | 阅读重点 |
|---|---|---|
| `electron/main.ts` | Electron 主进程入口 | 窗口、IPC、应用生命周期 |
| `electron/preload.ts` | Renderer 白名单 API | 为什么前端只能通过 `window.electronAPI` 访问桌面能力 |
| `electron/ipc/app-handlers.ts` | 应用级 IPC handlers | `get-backend`、首启 setup、analytics、models dir 更新 |
| `electron/python-backend.ts` | Python backend 守护器 | `spawn`、auth token、ready message、health/restart |
| `electron/python-setup.ts` | Python embed 下载器 | 首启下载、hash 校验、`python-next` 升级 |
| `electron/csp.ts` | Content Security Policy | Electron 安全策略 |
| `electron/path-validation.ts` | 文件路径白名单校验 | 导出/读写路径如何防止越界 |
| `electron/analytics.ts` | 匿名遥测实现 | telemetry 是如何发送与关闭的 |

## 4. Backend 必读文件

| 文件 | 职责 | 阅读重点 |
|---|---|---|
| `backend/ltx2_server.py` | 运行时组合根 | 设备检测、`RuntimeConfig`、Uvicorn 启动 |
| `backend/app_factory.py` | FastAPI app factory | auth middleware、异常处理、router 注册 |
| `backend/app_handler.py` | 依赖装配中心 | 各 handler 如何被组合进共享应用状态 |
| `backend/_routes/generation.py` | 生成路由 | 薄路由如何委派到 handler |
| `backend/handlers/video_generation_handler.py` | 视频生成主逻辑 | 本地/云端分流与进度上报 |
| `backend/handlers/image_generation_handler.py` | 图像生成逻辑 | ZIT 本地模式与 FAL API 模式 |
| `backend/handlers/retake_handler.py` | 视频 Retake | 局部重生成主链 |
| `backend/handlers/pipelines_handler.py` | pipeline 生命周期 | GPU pipeline 加载、切换与缓存 |
| `backend/handlers/generation_handler.py` | 生成状态机 | 进度、取消、完成、失败 |
| `backend/handlers/download_handler.py` | 模型下载与状态 | 模型获取与 staging 目录 |
| `backend/handlers/models_handler.py` | 模型状态查询 | required models 与 UI 展示对接 |

## 5. Runtime / State / Services 核心文件

| 文件 | 职责 | 阅读重点 |
|---|---|---|
| `backend/state/app_settings.py` | 设置 schema 与 patch model | API key、模型目录、local/API 选择 |
| `backend/state/app_state_types.py` | AppState 类型定义 | 显式状态机的组织方式 |
| `backend/runtime_config/runtime_policy.py` | 本地或 API 的判定逻辑 | `Darwin` 与 `VRAM < 31GB` 的处理 |
| `backend/runtime_config/model_download_specs.py` | 模型规格表 | 模型名、体积、repo_id、required models |
| `backend/services/ltx_api_client/ltx_api_client_impl.py` | LTX API 封装 | 上传、生成、下载、错误处理 |
| `backend/services/model_downloader/hugging_face_downloader.py` | Hugging Face 下载器 | 真实模型下载边界 |
| `backend/services/task_runner/threading_runner.py` | 后台线程任务 | 后台异常如何集中处理 |
| `backend/logging_policy.py` | request/background 日志策略 | 统一 traceback policy |

## 6. 测试与工程纪律

| 文件 | 作用 | 说明 |
|---|---|---|
| `backend/tests/test_no_mock_usage.py` | 禁止 mock/patch | 强制 backend 走 fake service 风格 |
| `backend/tests/test_pyright.py` | pyright 门禁 | 类型错误和 warning 都会拦住 |
| `AGENTS.md` | 仓库内开发规范 | 明确 frontend/backend 的约束与工作流 |

## 7. 最推荐的源码走读顺序

1. `README.md`
2. `backend/architecture.md`
3. `frontend/App.tsx`
4. `frontend/lib/backend.ts`
5. `electron/preload.ts`
6. `electron/python-backend.ts`
7. `backend/ltx2_server.py`
8. `backend/app_factory.py`
9. `frontend/hooks/use-generation.ts`
10. `backend/handlers/video_generation_handler.py`
11. `backend/state/app_settings.py`
12. `backend/runtime_config/model_download_specs.py`

这套顺序遵循的是：先看边界，再看启动，再看请求链，最后看推理和配置。
