from __future__ import annotations
import cv2
import numpy as np
import time
import os
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QDoubleSpinBox, QSpinBox, QGroupBox, QFormLayout, QMessageBox,
    QCheckBox, QPlainTextEdit
)
from ui.worker import AutomationWorker
from core.settings import load_settings, save_settings
from core.session_logger import default_log_dir

STYLE = """
QMainWindow, QWidget { background:#171a1f; color:#e9edf2; font-family:'Microsoft YaHei UI'; font-size:13px; }
QGroupBox { border:1px solid #343b45; border-radius:8px; margin-top:10px; padding:12px; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; }
QLineEdit, QDoubleSpinBox, QSpinBox { background:#242a32; border:1px solid #424b58; border-radius:5px; padding:6px; }
QPushButton { background:#303946; border:0; border-radius:6px; padding:9px 16px; }
QPushButton#start { background:#3b8c67; color:white; font-weight:700; }
QPushButton#stop { background:#a4474a; color:white; font-weight:700; }
QLabel#status { color:#a8c9ff; }
QPlainTextEdit { background:#101318; border:1px solid #343b45; border-radius:6px; color:#b8c5d2; }
QCheckBox { spacing:7px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动化小游戏程序 · 0.12.0")
        self.resize(980, 820)
        self.setMinimumSize(760, 650)
        self.setStyleSheet(STYLE)
        self.worker = None
        self._last_log_message = ""
        self._last_log_time = 0.0
        self.settings = load_settings()

        root_widget = QWidget()
        root = QVBoxLayout(root_widget)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel("渐隐图形实时跟踪 · 0.12.0")
        title.setStyleSheet("font-size:21px;font-weight:700;")
        title_row.addWidget(title)
        title_row.addStretch()
        self.log_dir_btn = QPushButton("打开日志目录")
        self.log_dir_btn.clicked.connect(self.open_log_dir)
        title_row.addWidget(self.log_dir_btn)
        root.addLayout(title_row)

        settings = QGroupBox("运行设置")
        form = QFormLayout(settings)
        self.window_title = QLineEdit(str(self.settings["window_title"]))
        self.fps = QDoubleSpinBox()
        self.fps.setRange(5, 120)
        self.fps.setDecimals(0)
        self.fps.setValue(float(self.settings["fps"]))
        self.fps.setSuffix(" FPS")
        self.max_lost = QSpinBox()
        self.max_lost.setRange(1, 30)
        self.max_lost.setValue(int(self.settings["max_lost_frames"]))
        self.acquire_frames = QSpinBox()
        self.acquire_frames.setRange(1, 10)
        self.acquire_frames.setValue(int(self.settings["acquire_frames"]))
        self.cursor_smoothing = QDoubleSpinBox()
        self.cursor_smoothing.setRange(0.0, 0.90)
        self.cursor_smoothing.setDecimals(2)
        self.cursor_smoothing.setSingleStep(0.05)
        self.cursor_smoothing.setValue(float(self.settings["cursor_smoothing"]))
        self.cursor_deadband = QDoubleSpinBox()
        self.cursor_deadband.setRange(0.0, 30.0)
        self.cursor_deadband.setDecimals(1)
        self.cursor_deadband.setSuffix(" px")
        self.cursor_deadband.setValue(float(self.settings["cursor_deadband_px"]))
        self.cursor_max_speed = QSpinBox()
        self.cursor_max_speed.setRange(0, 20000)
        self.cursor_max_speed.setSingleStep(250)
        self.cursor_max_speed.setSuffix(" px/s (0=不限)")
        self.cursor_max_speed.setValue(int(self.settings["cursor_max_speed_px_s"]))
        form.addRow("游戏窗口标题包含", self.window_title)
        form.addRow("捕获/跟踪频率", self.fps)
        form.addRow("连续丢失容忍帧数", self.max_lost)
        form.addRow("目标初始化确认帧数", self.acquire_frames)
        form.addRow("鼠标平滑系数（0=直接跟随）", self.cursor_smoothing)
        form.addRow("鼠标死区", self.cursor_deadband)
        form.addRow("鼠标最大速度", self.cursor_max_speed)

        options = QHBoxLayout()
        self.follow_mouse = QCheckBox("自动移动鼠标跟随目标")
        self.follow_mouse.setChecked(bool(self.settings["follow_mouse"]))
        self.enable_events = QCheckBox("模板识别成功/失败并等待下一轮")
        self.enable_events.setChecked(bool(self.settings["enable_events"]))
        self.enable_ocr = QCheckBox("启用 Tesseract OCR 兜底（可选）")
        self.enable_ocr.setChecked(bool(self.settings["enable_ocr"]))
        options.addWidget(self.follow_mouse)
        options.addWidget(self.enable_events)
        options.addWidget(self.enable_ocr)
        options.addStretch()
        form.addRow("自动化选项", options)
        root.addWidget(settings)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("开始监测")
        self.start_btn.setObjectName("start")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addStretch()
        root.addLayout(buttons)

        self.status = QLabel("待机。先确认窗口标题，再开始监测。")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.preview = QLabel("实时捕获预览")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(380)
        self.preview.setStyleSheet("background:#0c0f13;border:1px solid #343b45;border-radius:8px;")
        root.addWidget(self.preview, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(125)
        self.log.setPlaceholderText("运行日志：目标点、鼠标位置、CSRT 丢失次数、轮次事件")
        root.addWidget(self.log)

        note = QLabel(
            "0.12.0：CSRT 渐隐跟踪、可见目标重捕获、成功/失败模板识别、"
            "鼠标平滑/死区/限速、窗口失焦保护与自动重连、分阶段性能日志。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#9ca6b2;font-size:11px;")
        root.addWidget(note)
        self.setCentralWidget(root)

    def _current_settings(self):
        return {
            "window_title": self.window_title.text().strip(),
            "fps": self.fps.value(),
            "follow_mouse": self.follow_mouse.isChecked(),
            "enable_events": self.enable_events.isChecked(),
            "enable_ocr": self.enable_ocr.isChecked(),
            "max_lost_frames": self.max_lost.value(),
            "acquire_frames": self.acquire_frames.value(),
            "cursor_smoothing": self.cursor_smoothing.value(),
            "cursor_deadband_px": self.cursor_deadband.value(),
            "cursor_max_speed_px_s": self.cursor_max_speed.value(),
        }

    def persist_settings(self):
        try:
            save_settings(self._current_settings())
        except Exception as exc:
            self.log.appendPlainText(f"设置保存失败：{exc}")

    def start(self):
        title = self.window_title.text().strip()
        if not title:
            QMessageBox.warning(self, "缺少窗口标题", "请输入游戏窗口标题关键字。")
            return
        self.persist_settings()
        self.worker = AutomationWorker(
            title, self.fps.value(), self.follow_mouse.isChecked(),
            self.enable_ocr.isChecked(), self.max_lost.value(),
            self.acquire_frames.value(), self.enable_events.isChecked(), self,
            cursor_smoothing=self.cursor_smoothing.value(),
            cursor_deadband_px=self.cursor_deadband.value(),
            cursor_max_speed_px_s=self.cursor_max_speed.value(),
        )
        self.worker.status.connect(self.set_status)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self._set_controls_enabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _set_controls_enabled(self, enabled):
        for widget in (self.window_title, self.fps, self.max_lost, self.acquire_frames,
                       self.cursor_smoothing, self.cursor_deadband, self.cursor_max_speed,
                       self.follow_mouse, self.enable_events, self.enable_ocr):
            widget.setEnabled(enabled)

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.status.setText("正在停止……")

    def set_status(self, message):
        self.status.setText(message)
        now = time.monotonic()
        if message != self._last_log_message or now - self._last_log_time >= 0.7:
            self.log.appendPlainText(message)
            self._last_log_message = message
            self._last_log_time = now

    def show_frame(self, frame, result, event):
        if frame is None:
            return
        view = frame.copy()
        if result.cursor_trail:
            pts = list(result.cursor_trail)
            if len(pts) > 1:
                cv2.polylines(view, [np.array(pts, dtype="int32")],
                              False, (0, 220, 255), 2, cv2.LINE_AA)
        if result.bbox is not None:
            x, y, w, h = result.bbox
            color = (0, 220, 100) if result.valid else (0, 150, 255)
            cv2.rectangle(view, (x, y), (x+w, y+h), color, 2)
        if result.point is not None:
            cv2.circle(view, result.point, 5, (0, 0, 255), -1)
            cv2.circle(view, result.point, 10, (255, 255, 255), 1)
        if result.cursor_point is not None:
            cv2.circle(view, result.cursor_point, 9, (0, 255, 80), 2)
            cv2.circle(view, result.cursor_point, 2, (0, 255, 80), -1)
        label = f"{result.state} | {result.source} | lost={result.lost_frames}"
        if result.score is not None:
            label += f" | CSRT={result.score:.3f}"
        if result.cursor_error is not None:
            label += f" | err={result.cursor_error:.1f}px"
        cv2.putText(view, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (80, 240, 160) if result.valid else (40, 180, 255), 2)
        if event is not None and (event.event != "none" or event.score is not None):
            event_text = f"{event.event.upper()} {event.source} {event.score or 0:.3f}"
            cv2.putText(view, event_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(pix)

    def show_error(self, message):
        self.set_status("错误：" + message)

    def on_finished(self):
        self._set_controls_enabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def open_log_dir(self):
        path = default_log_dir()
        path.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(path))
            else:
                QMessageBox.information(self, "日志目录", str(path))
        except Exception as exc:
            QMessageBox.warning(self, "无法打开目录", f"{path}\n{exc}")

    def closeEvent(self, event):
        self.persist_settings()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        event.accept()