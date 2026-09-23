# MOT Vision Lab：YOLOv8 多目标跟踪交互系统

这是对原“YOLOv8 + Qt 智能交通监控”项目的可运行复现与二次开发版本。新入口 `mot_app.py` 不依赖旧版 PySide6/supervision 接口，使用仓库自带 YOLOv8 权重完成检测，并在 Kalman + Hungarian 基线上接入类别约束的纯运动 OC-SORT。

![交互窗口实测](outputs/ui_smoke_test.png)

## 已实现功能

- 图片、视频、本地摄像头三类输入；原始画面与跟踪结果并排显示。
- YOLOv8 目标检测，支持置信度和 NMS IoU 动态配置。
- 可切换 `OC-SORT / Kalman-Hungarian` 跟踪器，便于使用同一检测器做可视化对照。
- OC-SORT 包含方向一致性匹配（OCM）、最后观测重关联（OCR）、遮挡后重更新（ORU）和低置信度二阶段救回。
- 8 维常速度 Kalman 状态 `[cx, cy, w, h, vx, vy, vw, vh]`，使用类别约束 IoU 与 Hungarian 全局匹配。
- Track ID、运动轨迹、最大丢失帧与轨迹生命周期管理。
- 可调计数线和上下行累计计数。
- 点击检测框或目标表格锁定 ID；支持开始、暂停、继续和停止。
- 视频播放速度支持 `0.5× / 1.0× / 2.0× / 不限速`，可在运行中即时调整。
- 标注图片/MP4 与逐帧跟踪 CSV 导出。
- 高 DPI Windows 适配、Unicode 路径图片读写、GUI 后台线程推理。

![alt text](../opencv_project/image-1.png)


## 一键运行

当前机器已验证的解释器为 `D:\miniconda\envs\PJT_1\python.exe`，可直接执行：

```powershell
Set-Location F:\Team_porject\MOT-main
.\run_mot.ps1
```

如果当前终端提示符包含 `MINGW64`，说明使用的是 Git Bash，应执行：

```bash
./run_mot.sh
```

若在其他机器复现，推荐 Python 3.10：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-repro.txt
python mot_app.py
```

`torch` 的 CUDA 版本应按本机显卡和驱动单独选择；没有 NVIDIA GPU 时界面会自动使用 CPU。

## 操作说明

1. 启动后默认载入仓库内置 `bus.jpg`，点击“开始检测”即可验证完整链路。
2. “打开文件”选择图片或视频；“摄像头”输入设备编号。
3. 运行前可选择 `OC-SORT` 或 `Kalman-Hungarian`，并调整检测阈值、关联阈值、最大丢失帧和计数线位置。播放速度默认为 `1.0×`，跑性能测试时选择“不限速”。
4. 在跟踪画面点击目标框，或在下方目标表格选择一行，即可锁定 Track ID。
5. 勾选保存后，结果写入 `outputs/run_时间戳/`：
   - 图片输入：`result.jpg`
   - 视频/摄像头输入：`result.mp4`
   - 所有输入：`tracks.csv`

快捷键：`Space` 开始/暂停/继续，`Esc` 停止，`Ctrl+O` 打开文件。

## 处理链路

```text
图片 / 视频 / 摄像头
        │
        ▼
YOLOv8 检测（bbox / class / confidence）
        │
        ▼
OC-SORT：Kalman 预测 ── OCM/OCR ── 低分救回 ── ORU
        │
        ├── 匹配轨迹：Kalman 更新、延续 ID
        ├── 未匹配检测：创建新 ID
        └── 未匹配轨迹：超过 max_age 后删除
        │
        ▼
轨迹绘制 / 越线计数 / GUI 指标 / MP4 + CSV 导出
```

关键文件：

| 文件 | 作用 |
| --- | --- |
| `mot_core.py` | 检测器封装、Kalman Track、Hungarian 关联、计数与可视化 |
| `mot_app.py` | Tkinter 交互窗口、后台工作线程、输入输出与 ID 点选 |
| `benchmark_trackers.py` | 固定同一批检测结果，对比基线跟踪器与 OC-SORT |
| `tests/test_mot_core.py` | IoU、ID 连续性、类别约束和越线计数单元测试 |
| `run_mot.ps1` | Windows 一键启动脚本 |
| `requirements-repro.txt` | 新交互版本的最小依赖 |
| `main.py` | 原项目的旧 PySide6 入口，仅保留作参考 |

## 验证

### 1. 单元测试

```powershell
D:\miniconda\envs\PJT_1\python.exe -m unittest discover -s tests -v
```

当前结果：10/10 通过，覆盖 IoU、ID 延续、类别隔离、越线计数、OC-SORT 低分救回、低分禁止建轨与 ORU 遮挡恢复。

### 2. 跟踪器固定检测 A/B

```powershell
D:\miniconda\envs\PJT_1\python.exe benchmark_trackers.py --input F:\Opencv\opencv\sources\samples\data\vtest.avi --device 0 --output outputs\ocsort_ab_vtest.json
```

795 帧固定检测流实测中，OC-SORT 执行 111 次低分救回、45 次 ORU；可见轨迹记录从 6181 增至 6247，本轮跟踪耗时约从 1.4 ms/帧增至 2.4 ms/帧。该结果只证明机制已运行并增加了轨迹覆盖，由于 `vtest.avi` 没有跟踪真值，不能据此宣称 IDF1/IDSW 已提升。

### 3. 无窗口端到端测试

```powershell
.\run_mot.ps1 --smoke-test --device 0
```

2026-09-22 在内置 `bus.jpg` 上验证：模型成功加载，得到 6 个检测和 6 条初始轨迹，结果输出到 `outputs/smoke_test/result.jpg`。

### 4. GUI 视觉测试

```powershell
.\run_mot.ps1 --ui-smoke-test
```

该命令会启动窗口、处理内置示例、保存 `outputs/ui_smoke_test.png` 后自动退出。视觉验收确认了双画面、KPI 卡片、参数侧栏、目标表格和结果状态均完整显示。

> 性能说明：静态图的首帧推理包含模型初始化，不能作为视频稳态 FPS。简历中不要写“实时 30 FPS”等未经完整视频基准验证的数字。

## 简历与面试材料

- 可直接使用的项目描述：[docs/resume_project.md](docs/resume_project.md)
- 主张与证据边界：[docs/career-claim-ledger.json](docs/career-claim-ledger.json)

推荐项目名称：**面向智能交通的 YOLOv8 与 OC-SORT 实时多目标跟踪系统（开源复现与二次开发）**。

## 原项目与许可

原 README 演示链接：

- [Bilibili](https://www.bilibili.com/video/BV1yX4y1m7Fe/)
- [YouTube](https://youtu.be/_77LrsXaYzM)

原项目参考了 `Jai-wei/YOLOv8-PySide6-GUI`、Ultralytics YOLO 与 Qt for Python。仓库沿用 GPL-3.0 许可；公开发布或二次分发时应继续遵守该许可并保留归属说明。
