"""Interactive YOLOv8 multi-object tracking workbench.

Run ``python mot_app.py`` for the GUI or ``python mot_app.py --smoke-test``
for a headless end-to-end verification using the bundled sample image.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mot_core import FrameAnalyzer, FrameResult, ROOT, draw_tracking_overlay, read_image, write_image


MODEL_PATH = ROOT / "weights" / "轻量级模型.pt"
SAMPLE_IMAGE = ROOT / "ultralytics" / "assets" / "bus.jpg"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"}

PLAYBACK_SPEED_OPTIONS = {
    "0.5×": 0.5,
    "1.0×": 1.0,
    "2.0×": 2.0,
    "不限速": 0.0,
}


def target_frame_interval(source_fps: float, playback_speed: float) -> float:
    """返回视频每帧的目标播放间隔；0 表示不限速。"""
    if source_fps <= 0 or playback_speed <= 0:
        return 0.0
    return 1.0 / (source_fps * playback_speed)


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    value: str | int
    display_name: str


@dataclass
class FramePacket:
    original: np.ndarray
    result: FrameResult
    frame_index: int
    progress: float
    fps: float
    output_dir: str | None


class MOTWorkbench:
    COLORS = {
        "bg": "#0B1120",
        "panel": "#111A2E",
        "panel2": "#17233B",
        "text": "#E8EEF9",
        "muted": "#8EA0BE",
        "cyan": "#2DD4BF",
        "blue": "#38BDF8",
        "green": "#4ADE80",
        "orange": "#FBBF24",
        "red": "#FB7185",
        "line": "#263653",
    }

    def __init__(self, root, initial_source: str | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("MOT Vision Lab · YOLOv8 多目标跟踪")
        self.root.geometry("1480x900")
        self.root.minsize(1180, 720)
        self.root.configure(bg=self.COLORS["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.source: SourceSpec | None = None
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=2)
        self.event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.analyzer: FrameAnalyzer | None = None
        self.running = False
        self.selected_id: int | None = None
        self.latest_tracks = []
        self._canvas_meta: dict[str, tuple] = {}
        self._photos: dict[str, Any] = {}

        self.conf_var = tk.DoubleVar(value=0.35)
        self.nms_var = tk.DoubleVar(value=0.55)
        self.track_iou_var = tk.DoubleVar(value=0.25)
        self.max_age_var = tk.IntVar(value=20)
        self.line_var = tk.DoubleVar(value=0.62)
        self.traffic_only_var = tk.BooleanVar(value=True)
        self.save_output_var = tk.BooleanVar(value=True)
        self.device_var = tk.StringVar(value="auto")
        self.tracker_var = tk.StringVar(value="OC-SORT")
        self.subtitle_var = tk.StringVar(value="  YOLOv8 · OC-SORT · Hungarian")
        self.playback_speed_var = tk.StringVar(value="1.0×")
        self.playback_speed = PLAYBACK_SPEED_OPTIONS[self.playback_speed_var.get()]
        self.source_var = tk.StringVar(value="未选择输入源")
        self.status_var = tk.StringVar(value="请选择输入源，或直接运行内置示例")
        self.model_var = tk.StringVar(value=MODEL_PATH.name)
        self.target_var = tk.StringVar(value="0")
        self.fps_var = tk.StringVar(value="--")
        self.count_var = tk.StringVar(value="0")
        self.latency_var = tk.StringVar(value="--")
        self.selected_var = tk.StringVar(value="未选择")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._configure_style()
        self._build_layout()
        self._bind_shortcuts()
        self.root.after(30, self._poll_queues)

        selected = Path(initial_source) if initial_source else SAMPLE_IMAGE
        if selected.exists():
            self._set_file_source(selected)

    def _configure_style(self) -> None:
        style = self.ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.COLORS["bg"])
        style.configure("Panel.TFrame", background=self.COLORS["panel"])
        style.configure("Panel2.TFrame", background=self.COLORS["panel2"])
        style.configure("TLabel", background=self.COLORS["bg"], foreground=self.COLORS["text"], font=("Microsoft YaHei UI", 10))
        style.configure("Muted.TLabel", background=self.COLORS["panel"], foreground=self.COLORS["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("CardValue.TLabel", background=self.COLORS["panel2"], foreground=self.COLORS["text"], font=("Segoe UI", 22, "bold"))
        style.configure("CardName.TLabel", background=self.COLORS["panel2"], foreground=self.COLORS["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("TCheckbutton", background=self.COLORS["panel"], foreground=self.COLORS["text"], font=("Microsoft YaHei UI", 9))
        style.map("TCheckbutton", background=[("active", self.COLORS["panel"])])
        style.configure("TCombobox", fieldbackground=self.COLORS["panel2"], foreground=self.COLORS["text"])
        style.configure("Horizontal.TProgressbar", troughcolor=self.COLORS["panel2"], background=self.COLORS["cyan"], borderwidth=0)
        style.configure("Treeview", background=self.COLORS["panel2"], fieldbackground=self.COLORS["panel2"], foreground=self.COLORS["text"], rowheight=25, borderwidth=0)
        style.configure("Treeview.Heading", background=self.COLORS["panel"], foreground=self.COLORS["muted"], font=("Microsoft YaHei UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#185E70")])

    def _button(self, parent, text: str, command, color: str | None = None, width: int = 12):
        return self.tk.Button(
            parent, text=text, command=command, width=width, relief="flat", bd=0,
            bg=color or self.COLORS["panel2"], fg="#061017" if color else self.COLORS["text"],
            activebackground=color or "#223354", activeforeground="#061017" if color else self.COLORS["text"],
            font=("Microsoft YaHei UI", 10, "bold"), cursor="hand2", padx=10, pady=9,
        )

    def _build_layout(self) -> None:
        header = self.tk.Frame(self.root, bg=self.COLORS["bg"], height=72)
        header.pack(fill="x", padx=22, pady=(14, 8))
        self.tk.Label(header, text="MOT VISION LAB", bg=self.COLORS["bg"], fg=self.COLORS["cyan"],
                      font=("Segoe UI", 18, "bold")).pack(side="left")
        self.tk.Label(header, textvariable=self.subtitle_var, bg=self.COLORS["bg"], fg=self.COLORS["muted"],
                      font=("Segoe UI", 10)).pack(side="left", pady=(7, 0))
        self.tk.Label(header, textvariable=self.source_var, bg=self.COLORS["panel2"], fg=self.COLORS["text"],
                      font=("Microsoft YaHei UI", 9), padx=14, pady=8).pack(side="right")

        body = self.tk.Frame(self.root, bg=self.COLORS["bg"])
        body.pack(fill="both", expand=True, padx=22, pady=(0, 14))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, minsize=320)
        body.grid_rowconfigure(0, weight=1)

        content = self.tk.Frame(body, bg=self.COLORS["bg"])
        self.content_frame = content
        content.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        content.grid_columnconfigure((0, 1), weight=1)
        content.grid_rowconfigure(1, weight=1)

        cards = self.tk.Frame(content, bg=self.COLORS["bg"])
        cards.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        for column, (name, variable, accent) in enumerate([
            ("ACTIVE TARGETS", self.target_var, self.COLORS["blue"]),
            ("PROCESS FPS", self.fps_var, self.COLORS["green"]),
            ("LINE COUNT", self.count_var, self.COLORS["orange"]),
            ("INFERENCE", self.latency_var, self.COLORS["cyan"]),
        ]):
            cards.grid_columnconfigure(column, weight=1)
            card = self.tk.Frame(cards, bg=self.COLORS["panel2"], highlightbackground=self.COLORS["line"], highlightthickness=1)
            card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0 if column == 3 else 5))
            self.tk.Frame(card, bg=accent, height=3).pack(fill="x")
            self.ttk.Label(card, text=name, style="CardName.TLabel").pack(anchor="w", padx=14, pady=(10, 0))
            self.ttk.Label(card, textvariable=variable, style="CardValue.TLabel").pack(anchor="w", padx=14, pady=(0, 10))

        self.input_canvas = self._video_panel(content, "原始画面 / INPUT", 0)
        self.result_canvas = self._video_panel(content, "跟踪结果 / TRACKING", 1)
        self.result_canvas.bind("<Button-1>", self._select_track_at)

        table_frame = self.tk.Frame(content, bg=self.COLORS["panel"], height=155)
        table_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.tk.Label(table_frame, text="实时目标列表  ·  点击右侧画面可锁定 ID", bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                      font=("Microsoft YaHei UI", 9)).pack(anchor="w", padx=12, pady=(8, 4))
        self.track_table = self.ttk.Treeview(table_frame, columns=("id", "class", "confidence", "age", "center"), show="headings", height=4)
        headings = [("id", "ID", 65), ("class", "类别", 130), ("confidence", "置信度", 90), ("age", "存活帧", 80), ("center", "中心坐标", 130)]
        for key, title, width in headings:
            self.track_table.heading(key, text=title)
            self.track_table.column(key, width=width, anchor="center")
        self.track_table.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.track_table.bind("<<TreeviewSelect>>", self._select_track_from_table)

        sidebar = self.tk.Frame(body, bg=self.COLORS["panel"], width=320)
        self.sidebar_frame = sidebar
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_propagate(False)
        self._build_sidebar(sidebar)

        footer = self.tk.Frame(self.root, bg=self.COLORS["panel"], height=42)
        footer.pack(fill="x", side="bottom")
        self.tk.Label(footer, textvariable=self.status_var, bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                      font=("Microsoft YaHei UI", 9)).pack(side="left", padx=22)
        self.ttk.Progressbar(footer, variable=self.progress_var, maximum=100, length=260).pack(side="right", padx=22, pady=12)

    def _video_panel(self, parent, title: str, column: int):
        panel = self.tk.Frame(parent, bg=self.COLORS["panel"], highlightbackground=self.COLORS["line"], highlightthickness=1)
        panel.grid(row=1, column=column, sticky="nsew", padx=(0, 5) if column == 0 else (5, 0))
        self.tk.Label(panel, text=title, bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                      font=("Segoe UI", 9, "bold"), padx=12, pady=9).pack(anchor="w")
        canvas = self.tk.Canvas(panel, bg="#050913", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        canvas.bind("<Configure>", lambda _event: self._redraw_latest())
        return canvas

    def _build_sidebar(self, parent) -> None:
        self.tk.Label(parent, text="CONTROL CENTER", bg=self.COLORS["panel"], fg=self.COLORS["text"],
                      font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=18, pady=(18, 10))
        source_row = self.tk.Frame(parent, bg=self.COLORS["panel"])
        source_row.pack(fill="x", padx=16)
        self._button(source_row, "打开文件", self._open_file, self.COLORS["cyan"], 11).pack(side="left", expand=True, fill="x", padx=(0, 5))
        self._button(source_row, "摄像头", self._open_camera, None, 10).pack(side="left", expand=True, fill="x", padx=(5, 0))

        self._separator(parent)
        self._section_title(parent, "检测参数")
        self._slider(parent, "置信度阈值", self.conf_var, 0.10, 0.90, 0.01)
        self._slider(parent, "NMS IoU", self.nms_var, 0.20, 0.90, 0.01)
        self._slider(parent, "跟踪关联 IoU", self.track_iou_var, 0.05, 0.70, 0.01)
        self._slider(parent, "计数线位置", self.line_var, 0.30, 0.80, 0.01)

        compact = self.tk.Frame(parent, bg=self.COLORS["panel"])
        compact.pack(fill="x", padx=18, pady=(7, 2))
        self.tk.Label(compact, text="最大丢失帧", bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                      font=("Microsoft YaHei UI", 9)).pack(side="left")
        spin = self.tk.Spinbox(compact, from_=1, to=90, textvariable=self.max_age_var, width=6,
                               bg=self.COLORS["panel2"], fg=self.COLORS["text"], buttonbackground=self.COLORS["panel2"],
                               insertbackground=self.COLORS["text"], relief="flat")
        spin.pack(side="right")

        tracker_row = self.tk.Frame(parent, bg=self.COLORS["panel"])
        tracker_row.pack(fill="x", padx=18, pady=7)
        self.tk.Label(tracker_row, text="跟踪器", bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                      font=("Microsoft YaHei UI", 9)).pack(side="left")
        tracker_combo = self.ttk.Combobox(
            tracker_row,
            textvariable=self.tracker_var,
            values=("OC-SORT", "Kalman-Hungarian"),
            width=17,
            state="readonly",
        )
        tracker_combo.pack(side="right")
        tracker_combo.bind("<<ComboboxSelected>>", self._on_tracker_changed)

        device_row = self.tk.Frame(parent, bg=self.COLORS["panel"])
        device_row.pack(fill="x", padx=18, pady=7)
        self.tk.Label(device_row, text="推理设备", bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                      font=("Microsoft YaHei UI", 9)).pack(side="left")
        combo = self.ttk.Combobox(device_row, textvariable=self.device_var, values=("auto", "cpu", "0"), width=8, state="readonly")
        combo.pack(side="right")

        speed_row = self.tk.Frame(parent, bg=self.COLORS["panel"])
        speed_row.pack(fill="x", padx=18, pady=7)
        self.tk.Label(speed_row, text="播放速度", bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                      font=("Microsoft YaHei UI", 9)).pack(side="left")
        speed_combo = self.ttk.Combobox(
            speed_row,
            textvariable=self.playback_speed_var,
            values=tuple(PLAYBACK_SPEED_OPTIONS),
            width=8,
            state="readonly",
        )
        speed_combo.pack(side="right")
        speed_combo.bind("<<ComboboxSelected>>", self._on_playback_speed_changed)

        self.ttk.Checkbutton(parent, text="仅显示交通参与者（人/车）", variable=self.traffic_only_var).pack(anchor="w", padx=18, pady=4)
        self.ttk.Checkbutton(parent, text="保存标注视频/图片与 CSV", variable=self.save_output_var).pack(anchor="w", padx=18, pady=4)

        self._separator(parent)
        self._section_title(parent, "运行状态")
        self._info_row(parent, "模型", self.model_var)
        self._info_row(parent, "锁定目标", self.selected_var)

        action_row = self.tk.Frame(parent, bg=self.COLORS["panel"])
        action_row.pack(fill="x", side="bottom", padx=16, pady=16)
        self.start_button = self._button(action_row, "开始检测", self._start_or_pause, self.COLORS["green"], 13)
        self.start_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self._button(action_row, "停止", self._stop, self.COLORS["red"], 9).pack(side="left", expand=True, fill="x", padx=(5, 0))

        tips = self.tk.Label(parent, text="快捷键  Space 开始/暂停  ·  Esc 停止\n点击检测框或目标列表可锁定 ID",
                             bg=self.COLORS["panel"], fg=self.COLORS["muted"], justify="left",
                             font=("Microsoft YaHei UI", 8), wraplength=280)
        tips.pack(side="bottom", anchor="w", padx=18, pady=(0, 4))

    def _section_title(self, parent, text: str) -> None:
        self.tk.Label(parent, text=text, bg=self.COLORS["panel"], fg=self.COLORS["text"],
                      font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w", padx=18, pady=(0, 7))

    def _separator(self, parent) -> None:
        self.tk.Frame(parent, bg=self.COLORS["line"], height=1).pack(fill="x", padx=18, pady=15)

    def _slider(self, parent, label: str, variable, start: float, end: float, resolution: float) -> None:
        row = self.tk.Frame(parent, bg=self.COLORS["panel"])
        row.pack(fill="x", padx=18, pady=3)
        value = self.tk.Label(row, text=f"{variable.get():.2f}", bg=self.COLORS["panel"], fg=self.COLORS["cyan"],
                              font=("Consolas", 9, "bold"), width=5)
        self.tk.Label(row, text=label, bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                      font=("Microsoft YaHei UI", 9)).pack(side="left")
        value.pack(side="right")
        scale = self.tk.Scale(parent, from_=start, to=end, resolution=resolution, orient="horizontal", variable=variable,
                              showvalue=False, bg=self.COLORS["panel"], fg=self.COLORS["text"],
                              troughcolor=self.COLORS["panel2"], activebackground=self.COLORS["cyan"],
                              highlightthickness=0, bd=0, sliderlength=16,
                              command=lambda raw, target=value: target.configure(text=f"{float(raw):.2f}"))
        scale.pack(fill="x", padx=18)

    def _info_row(self, parent, label: str, variable) -> None:
        row = self.tk.Frame(parent, bg=self.COLORS["panel"])
        row.pack(fill="x", padx=18, pady=5)
        self.tk.Label(row, text=label, bg=self.COLORS["panel"], fg=self.COLORS["muted"], font=("Microsoft YaHei UI", 9)).pack(side="left")
        self.tk.Label(row, textvariable=variable, bg=self.COLORS["panel"], fg=self.COLORS["text"],
                      font=("Microsoft YaHei UI", 9, "bold"), wraplength=170).pack(side="right")

    def _bind_shortcuts(self) -> None:
        self.root.bind("<space>", lambda _event: self._start_or_pause())
        self.root.bind("<Escape>", lambda _event: self._stop())
        self.root.bind("<Control-o>", lambda _event: self._open_file())

    def _open_file(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="选择图片或视频",
            filetypes=[("媒体文件", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mkv *.mov *.flv *.wmv"), ("所有文件", "*.*")],
        )
        if path:
            self._set_file_source(Path(path))

    def _set_file_source(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            self.status_var.set(f"不支持的文件类型：{suffix}")
            return
        self._stop()
        kind = "image" if suffix in IMAGE_SUFFIXES else "video"
        self.source = SourceSpec(kind, str(path), path.name)
        self.source_var.set(path.name)
        self.selected_id = None
        self.selected_var.set("未选择")
        if kind == "image":
            frame = read_image(path)
        else:
            capture = cv2.VideoCapture(str(path))
            ok, frame = capture.read()
            capture.release()
            frame = frame if ok else None
        if frame is None:
            self.status_var.set("输入源读取失败")
            return
        self._display_frame(self.input_canvas, frame, "input")
        self._display_frame(self.result_canvas, frame, "result")
        self.status_var.set(f"已加载{('图片' if kind == 'image' else '视频')}：{path.name}")

    def _open_camera(self) -> None:
        from tkinter import simpledialog

        index = simpledialog.askinteger("选择摄像头", "请输入摄像头编号：", initialvalue=0, minvalue=0, maxvalue=16)
        if index is not None:
            self._stop()
            self.source = SourceSpec("camera", index, f"Camera {index}")
            self.source_var.set(f"Camera {index}")
            self.status_var.set(f"已选择摄像头 {index}，点击开始检测")

    def _start_or_pause(self) -> None:
        if self.running:
            if self.pause_event.is_set():
                self.pause_event.clear()
                self.start_button.configure(text="暂停检测", bg=self.COLORS["orange"])
                self.status_var.set("继续检测")
            else:
                self.pause_event.set()
                self.start_button.configure(text="继续检测", bg=self.COLORS["green"])
                self.status_var.set("检测已暂停")
            return
        if self.source is None:
            self.status_var.set("请先选择输入源")
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.running = True
        self.progress_var.set(0)
        self.start_button.configure(text="暂停检测", bg=self.COLORS["orange"])
        config = {
            "confidence": float(self.conf_var.get()),
            "nms_iou": float(self.nms_var.get()),
            "track_iou": float(self.track_iou_var.get()),
            "max_age": int(self.max_age_var.get()),
            "line_ratio": float(self.line_var.get()),
            "traffic_only": bool(self.traffic_only_var.get()),
            "save_output": bool(self.save_output_var.get()),
            "device": self.device_var.get(),
            "tracker_type": "ocsort" if self.tracker_var.get() == "OC-SORT" else "sort",
        }
        self.worker = threading.Thread(target=self._worker_loop, args=(self.source, config), daemon=True)
        self.worker.start()

    def _on_playback_speed_changed(self, _event=None) -> None:
        label = self.playback_speed_var.get()
        self.playback_speed = PLAYBACK_SPEED_OPTIONS.get(label, 1.0)
        if self.source is not None and self.source.kind == "video":
            self.status_var.set(f"播放速度已调整为 {label}")

    def _on_tracker_changed(self, _event=None) -> None:
        tracker_name = self.tracker_var.get()
        self.subtitle_var.set(f"  YOLOv8 · {tracker_name} · Hungarian")
        suffix = "；将在下次开始时生效" if self.running else ""
        self.status_var.set(f"已选择跟踪器 {tracker_name}{suffix}")

    def _stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        if self.running:
            self.status_var.set("正在停止当前任务…")

    def _worker_loop(self, source: SourceSpec, config: dict[str, Any]) -> None:
        capture = None
        writer = None
        csv_file = None
        output_dir: Path | None = None
        try:
            self.event_queue.put(("status", "正在加载 YOLOv8 模型与跟踪器…"))
            self.analyzer = FrameAnalyzer(
                MODEL_PATH,
                confidence=config["confidence"], nms_iou=config["nms_iou"],
                track_iou=config["track_iou"], max_age=config["max_age"],
                line_ratio=config["line_ratio"], traffic_only=config["traffic_only"],
                device=config["device"],
                tracker_type=config["tracker_type"],
            )
            self.analyzer.reset()
            if config["save_output"]:
                output_dir = ROOT / "outputs" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
                output_dir.mkdir(parents=True, exist_ok=True)
                csv_file = (output_dir / "tracks.csv").open("w", newline="", encoding="utf-8-sig")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["frame", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
            else:
                csv_writer = None

            if source.kind == "image":
                frame = read_image(source.value)
                if frame is None:
                    raise RuntimeError("无法读取图片")
                total_frames = 1
                fps_source = 1.0
            else:
                capture = cv2.VideoCapture(source.value)
                if not capture.isOpened():
                    raise RuntimeError("无法打开视频或摄像头")
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if source.kind == "video" else 0
                fps_source = capture.get(cv2.CAP_PROP_FPS) or 25.0

            frame_index = 0
            fps_ema = 0.0
            self.event_queue.put(("status", "检测运行中：点击跟踪框可锁定目标 ID"))
            while not self.stop_event.is_set():
                while self.pause_event.is_set() and not self.stop_event.is_set():
                    time.sleep(0.05)
                if self.stop_event.is_set():
                    break
                if source.kind == "image":
                    if frame_index > 0:
                        break
                    current = frame.copy()
                else:
                    ok, current = capture.read()
                    if not ok:
                        break
                frame_index += 1
                started = time.perf_counter()
                result = self.analyzer.process(current, self.selected_id)
                elapsed = max(time.perf_counter() - started, 1e-6)
                instant_fps = 1.0 / elapsed
                fps_ema = instant_fps if fps_ema == 0 else 0.85 * fps_ema + 0.15 * instant_fps

                if output_dir is not None:
                    if source.kind == "image":
                        write_image(output_dir / "result.jpg", result.frame)
                    else:
                        if writer is None:
                            height, width = result.frame.shape[:2]
                            writer = cv2.VideoWriter(str(output_dir / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_source, (width, height))
                        writer.write(result.frame)
                    if csv_writer is not None:
                        for track in result.tracks:
                            x1, y1, x2, y2 = track.xyxy.tolist()
                            name = self.analyzer.detector.names[track.class_id]
                            csv_writer.writerow([frame_index, track.track_id, track.class_id, name, f"{track.score:.4f}",
                                                 f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"])
                progress = 100.0 if source.kind == "image" else (100.0 * frame_index / total_frames if total_frames > 0 else 0.0)
                packet = FramePacket(current, result, frame_index, progress, fps_ema, str(output_dir) if output_dir else None)
                try:
                    self.frame_queue.put_nowait(packet)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(packet)
                if source.kind == "video":
                    interval = target_frame_interval(fps_source, self.playback_speed)
                    remaining = interval - (time.perf_counter() - started)
                    if remaining > 0:
                        self.stop_event.wait(remaining)
            completed = not self.stop_event.is_set()
            self.event_queue.put(("done", {"completed": completed, "output_dir": str(output_dir) if output_dir else None,
                                            "frames": frame_index}))
        except Exception as exc:
            self.event_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()

    def _poll_queues(self) -> None:
        newest = None
        try:
            while True:
                newest = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        if newest is not None:
            self._handle_frame(newest)
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "error":
                    self.running = False
                    self.start_button.configure(text="开始检测", bg=self.COLORS["green"])
                    self.status_var.set(f"运行失败：{payload}")
                elif kind == "done":
                    self.running = False
                    self.start_button.configure(text="开始检测", bg=self.COLORS["green"])
                    if payload["completed"]:
                        suffix = f"，结果：{payload['output_dir']}" if payload["output_dir"] else ""
                        self.status_var.set(f"处理完成，共 {payload['frames']} 帧{suffix}")
                    else:
                        self.status_var.set("任务已停止")
        except queue.Empty:
            pass
        self.root.after(30, self._poll_queues)

    def _handle_frame(self, packet: FramePacket) -> None:
        self.latest_tracks = packet.result.tracks
        self._display_frame(self.input_canvas, packet.original, "input")
        self._display_frame(self.result_canvas, packet.result.frame, "result", packet.result.tracks)
        self.target_var.set(str(len(packet.result.tracks)))
        self.fps_var.set(f"{packet.fps:.1f}")
        self.count_var.set(str(packet.result.up_count + packet.result.down_count))
        self.latency_var.set(f"{packet.result.inference_ms:.0f} ms")
        self.progress_var.set(packet.progress)
        self._update_table(packet.result.tracks)

    def _display_frame(self, canvas, frame: np.ndarray, key: str, tracks=None) -> None:
        from PIL import Image, ImageTk

        canvas.update_idletasks()
        canvas_width = max(40, canvas.winfo_width())
        canvas_height = max(40, canvas.winfo_height())
        frame_height, frame_width = frame.shape[:2]
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        draw_width = max(1, int(frame_width * scale))
        draw_height = max(1, int(frame_height * scale))
        resized = cv2.resize(frame, (draw_width, draw_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        offset_x = (canvas_width - draw_width) // 2
        offset_y = (canvas_height - draw_height) // 2
        canvas.delete("all")
        canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)
        self._photos[key] = photo
        self._canvas_meta[key] = (offset_x, offset_y, scale, frame_width, frame_height, tracks or [])
        if key == "input":
            self._last_input = frame.copy()
        else:
            self._last_result = frame.copy()

    def _redraw_latest(self) -> None:
        if hasattr(self, "_last_input"):
            self._display_frame(self.input_canvas, self._last_input, "input")
        if hasattr(self, "_last_result"):
            self._display_frame(self.result_canvas, self._last_result, "result", self.latest_tracks)

    def _update_table(self, tracks) -> None:
        selected = self.selected_id
        self.track_table.delete(*self.track_table.get_children())
        names = self.analyzer.detector.names if self.analyzer else {}
        selected_item = None
        for track in tracks:
            class_name = names[track.class_id] if names else str(track.class_id)
            item = self.track_table.insert("", "end", values=(track.track_id, class_name, f"{track.score:.2f}", track.age,
                                                                  f"{track.center[0]}, {track.center[1]}"))
            if track.track_id == selected:
                selected_item = item
        if selected_item:
            self.track_table.selection_set(selected_item)

    def _select_track_at(self, event) -> None:
        meta = self._canvas_meta.get("result")
        if not meta:
            return
        offset_x, offset_y, scale, _width, _height, tracks = meta
        x = (event.x - offset_x) / scale
        y = (event.y - offset_y) / scale
        candidates = []
        for track in tracks:
            x1, y1, x2, y2 = track.xyxy
            if x1 <= x <= x2 and y1 <= y <= y2:
                candidates.append(((x2 - x1) * (y2 - y1), track.track_id))
        if candidates:
            self._set_selected_id(min(candidates)[1])

    def _select_track_from_table(self, _event) -> None:
        selection = self.track_table.selection()
        if selection:
            values = self.track_table.item(selection[0], "values")
            if values:
                self._set_selected_id(int(values[0]))

    def _set_selected_id(self, track_id: int) -> None:
        self.selected_id = track_id
        self.selected_var.set(f"ID {track_id}")
        self.status_var.set(f"已锁定目标 ID {track_id}；再次点击其他框可切换")
        self._update_table(self.latest_tracks)
        if self.analyzer is not None and hasattr(self, "_last_input"):
            highlighted = draw_tracking_overlay(
                self._last_input,
                self.latest_tracks,
                self.analyzer.detector.names,
                self.analyzer.counter,
                selected_id=track_id,
            )
            self._display_frame(self.result_canvas, highlighted, "result", self.latest_tracks)

    def _on_close(self) -> None:
        self.stop_event.set()
        self.root.after(80, self.root.destroy)


def run_smoke_test(device: str = "auto") -> int:
    output_dir = ROOT / "outputs" / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(SAMPLE_IMAGE)
    if image is None:
        raise RuntimeError(f"Cannot read sample image: {SAMPLE_IMAGE}")
    analyzer = FrameAnalyzer(MODEL_PATH, confidence=0.25, traffic_only=False, device=device)
    result = analyzer.process(image)
    output_path = output_dir / "result.jpg"
    if not write_image(output_path, result.frame):
        raise RuntimeError(f"Cannot write result: {output_path}")
    summary = {
        "status": "ok",
        "model": str(MODEL_PATH),
        "device": str(analyzer.detector.device),
        "detections": result.detections,
        "tracks": len(result.tracks),
        "inference_ms": round(result.inference_ms, 2),
        "output": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive YOLOv8 multi-object tracking workbench")
    parser.add_argument("--source", help="Initial image/video path")
    parser.add_argument("--smoke-test", action="store_true", help="Run one headless frame and save the result")
    parser.add_argument("--ui-smoke-test", action="store_true", help="Open, capture and close the GUI automatically")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "0"), help="Inference device for smoke test")
    args = parser.parse_args()
    if args.smoke_test:
        return run_smoke_test(args.device)

    if sys.platform == "win32":
        # Keep Tk geometry and screenshot coordinates in the same pixel space on
        # 125%/150% scaled displays, and make text/canvas rendering sharper.
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass

    import tkinter as tk

    root = tk.Tk()
    app = MOTWorkbench(root, args.source)
    if args.ui_smoke_test:
        from PIL import Image, ImageGrab

        screenshot_path = ROOT / "outputs" / "ui_smoke_test.png"

        def capture_window() -> None:
            root.deiconify()
            root.lift()
            root.attributes("-topmost", True)
            root.update_idletasks()
            root.update()
            left, top = root.winfo_rootx(), root.winfo_rooty()
            width, height = root.winfo_width(), root.winfo_height()
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            captured = False
            if sys.platform == "win32":
                try:
                    import ctypes
                    from ctypes import wintypes

                    class BitmapInfoHeader(ctypes.Structure):
                        _fields_ = [
                            ("biSize", wintypes.DWORD),
                            ("biWidth", wintypes.LONG),
                            ("biHeight", wintypes.LONG),
                            ("biPlanes", wintypes.WORD),
                            ("biBitCount", wintypes.WORD),
                            ("biCompression", wintypes.DWORD),
                            ("biSizeImage", wintypes.DWORD),
                            ("biXPelsPerMeter", wintypes.LONG),
                            ("biYPelsPerMeter", wintypes.LONG),
                            ("biClrUsed", wintypes.DWORD),
                            ("biClrImportant", wintypes.DWORD),
                        ]

                    class BitmapInfo(ctypes.Structure):
                        _fields_ = [("bmiHeader", BitmapInfoHeader),
                                    ("bmiColors", wintypes.DWORD * 3)]

                    user32 = ctypes.windll.user32
                    gdi32 = ctypes.windll.gdi32
                    hwnd = root.winfo_id()
                    window_dc = user32.GetWindowDC(hwnd)
                    memory_dc = gdi32.CreateCompatibleDC(window_dc)
                    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
                    previous = gdi32.SelectObject(memory_dc, bitmap)
                    if user32.PrintWindow(hwnd, memory_dc, 2):
                        info = BitmapInfo()
                        info.bmiHeader.biSize = ctypes.sizeof(BitmapInfoHeader)
                        info.bmiHeader.biWidth = width
                        info.bmiHeader.biHeight = -height
                        info.bmiHeader.biPlanes = 1
                        info.bmiHeader.biBitCount = 32
                        buffer = ctypes.create_string_buffer(width * height * 4)
                        if gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer,
                                          ctypes.byref(info), 0):
                            Image.frombuffer("RGB", (width, height), buffer.raw,
                                             "raw", "BGRX", 0, 1).save(screenshot_path)
                            captured = True
                    gdi32.SelectObject(memory_dc, previous)
                    gdi32.DeleteObject(bitmap)
                    gdi32.DeleteDC(memory_dc)
                    user32.ReleaseDC(hwnd, window_dc)
                except (AttributeError, OSError, ValueError):
                    captured = False
            if not captured:
                ImageGrab.grab((left, top, left + width, top + height)).save(screenshot_path)
            root.attributes("-topmost", False)
            print(json.dumps({
                "status": "ok",
                "ui_screenshot": str(screenshot_path),
                "window": [width, height],
                "content": [app.content_frame.winfo_x(), app.content_frame.winfo_width()],
                "sidebar": [app.sidebar_frame.winfo_x(), app.sidebar_frame.winfo_width()],
            }, ensure_ascii=False))

        root.after(400, app._start_or_pause)
        root.after(5000, capture_window)
        root.after(5400, app._on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
