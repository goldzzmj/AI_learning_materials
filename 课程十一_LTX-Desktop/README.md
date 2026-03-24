# 课程十一：LTX-Desktop 调研资料

> 项目仓库：`https://github.com/Lightricks/LTX-Desktop`
>
> 项目定位：一个把 LTX 视频模型封装成桌面应用的工程化系统，采用 `React + Electron + FastAPI + PyTorch` 的三层架构。

## 你会在这份资料里学到什么

- LTX Desktop 为什么不是“前端直接调模型”，而是 `Renderer -> Electron -> Python Backend` 三段式架构。
- 本地 GPU 推理与 API-only 模式如何在同一套产品里共存。
- Electron 桌面壳、Python 嵌入式环境、模型下载、认证、日志、遥测、打包发布是如何被串起来的。
- 如果你要做一个 AI 桌面应用，这个仓库有哪些值得复用的工程模式。

## 文档导航

| 文件 | 作用 | 建议阅读顺序 |
|---|---|---|
| `LTX-Desktop技术深度解析.md` | 主文档，系统讲清架构、调用链、核心源码与工程设计 | 1 |
| `LTX-Desktop关键文件索引.md` | 按前端 / Electron / 后端整理重点文件与阅读理由 | 2 |
| `LTX-Desktop面试问答.md` | 面试或技术复盘时可直接复用的高频问题与答案 | 3 |

## 一句话结论

LTX Desktop 的关键价值，不只是“能生成视频”，而是把高门槛的视频生成模型包装成了一个可安装、可升级、可回退到 API、可做项目管理和视频编辑的桌面产品。

## 本次调研重点参考的仓库资料

- `README.md`
- `backend/architecture.md`
- `docs/INSTALLER.md`
- `docs/TELEMETRY.md`
- `docs/CONTRIBUTING.md`
- `frontend/lib/backend.ts`
- `electron/preload.ts`
- `electron/python-backend.ts`
- `electron/python-setup.ts`
- `backend/ltx2_server.py`
- `backend/app_factory.py`
- `backend/handlers/video_generation_handler.py`
- `backend/runtime_config/runtime_policy.py`
- `backend/state/app_settings.py`
- `backend/runtime_config/model_download_specs.py`

## 推荐使用方式

1. 先读 `LTX-Desktop技术深度解析.md` 建立全局理解。
2. 再用 `LTX-Desktop关键文件索引.md` 对照源码做走读。
3. 最后用 `LTX-Desktop面试问答.md` 检查自己是否真正理解了设计取舍。
