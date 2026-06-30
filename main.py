"""Whisper 字幕生成器（PyQt5 GUI）。

- Apple Silicon 使用 mlx_whisper（MLX 优化），其余平台使用 openai-whisper(+torch)。
- ffmpeg 由 imageio-ffmpeg 随包提供：启动时注入 PATH，使本程序与 whisper /
  mlx_whisper 内部的 `ffmpeg` 调用都能找到它，实现开箱即用。
- 模型在首次使用时按需下载（不随包封装）。
- 支持多文件批量、语言/任务选择、模型缓存、确定性进度（尽力而为）、
  以及可选导出 Apple .itt。
"""
import os
import re
import sys
import shutil
import tempfile
import platform
import time

from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QWidget, QComboBox, QProgressBar, QCheckBox, QFrame, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon

import srt2itt
import downloader

# 支持的音频与视频扩展名（基于 ffmpeg 常见可解码格式）
SUPPORTED_AUDIO_EXTENSIONS = {
    '.aac', '.aiff', '.alac', '.amr', '.flac', '.m4a', '.mp3',
    '.ogg', '.opus', '.wav', '.wma',
}
SUPPORTED_VIDEO_EXTENSIONS = {
    '.avi', '.flv', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg',
    '.mpg', '.ts', '.webm', '.wmv',
}
SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS.union(SUPPORTED_VIDEO_EXTENSIONS)

# 语言下拉项：(whisper 语言码或 None, 显示名)
LANGUAGES = [
    (None, '自动检测'),
    ('zh', '中文'),
    ('en', '英语'),
    ('ja', '日语'),
    ('ko', '韩语'),
    ('yue', '粤语'),
    ('es', '西班牙语'),
    ('fr', '法语'),
    ('de', '德语'),
    ('ru', '俄语'),
    ('it', '意大利语'),
    ('pt', '葡萄牙语'),
    ('ar', '阿拉伯语'),
    ('hi', '印地语'),
    ('th', '泰语'),
    ('vi', '越南语'),
]

# 模型注册表：id, 显示名, MLX 仓库(Apple Silicon), whisper 名(其余平台), 约大小(MB)
# 注意 turbo 系列仓库名不遵循 whisper-{size}-mlx 规则，需在此显式映射。
MODELS = [
    ('large-v3-turbo', '⚡ Large V3 Turbo（推荐 · 又快又准）',
     'mlx-community/whisper-large-v3-turbo', 'large-v3-turbo', 1600),
    ('large-v3', 'Large V3（最高准确度 · 较慢）',
     'mlx-community/whisper-large-v3-mlx', 'large-v3', 3100),
    ('medium', 'Medium（中型）',
     'mlx-community/whisper-medium-mlx', 'medium', 1500),
    ('small', 'Small（小型）',
     'mlx-community/whisper-small-mlx', 'small', 480),
    ('base', 'Base（基础）',
     'mlx-community/whisper-base-mlx', 'base', 145),
    ('tiny', 'Tiny（最快 · 准确度低）',
     'mlx-community/whisper-tiny-mlx', 'tiny', 75),
]

_MODEL_BY_ID = {m[0]: m for m in MODELS}


def model_mlx_repo(model_id):
    return _MODEL_BY_ID[model_id][2]


def model_whisper_name(model_id):
    return _MODEL_BY_ID[model_id][3]


def model_approx_mb(model_id):
    return _MODEL_BY_ID[model_id][4]


def is_apple_silicon():
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'


def resource_path(rel):
    """解析随包资源路径（兼容 PyInstaller 冻结环境）。"""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


# ----------------------------- ffmpeg 注入 -----------------------------

_FFMPEG_PATH = None


def setup_ffmpeg():
    """把随包的 ffmpeg 暴露为 PATH 中名为 `ffmpeg` 的可执行文件。

    imageio-ffmpeg 自带的二进制名形如 `ffmpeg-macos-arm64-vX`，而 whisper /
    mlx_whisper 内部以 `ffmpeg` 调用子进程，故把它复制成一个名为 ffmpeg 的真实文件
    放到稳定的缓存目录，并把该目录前置到 PATH。

    用「复制」而非「软链」很关键：macOS 的 App Translocation 会让 .app 每次从不同
    的随机临时路径运行，旧的软链会指向已消失的路径而失效，导致回退后 PATH 里只有名为
    `ffmpeg-macos-...` 的二进制，whisper 以 `ffmpeg` 调用时报 “No such file”。复制出
    的真实文件不受其影响；按大小判断复用，仅首次复制。
    """
    global _FFMPEG_PATH
    src = None
    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        src = None
    if not src or not os.path.exists(src):
        _FFMPEG_PATH = shutil.which('ffmpeg')
        if _FFMPEG_PATH:
            os.environ['PATH'] = (os.path.dirname(_FFMPEG_PATH) + os.pathsep
                                  + os.environ.get('PATH', ''))
        return _FFMPEG_PATH

    bindir = os.path.join(os.path.expanduser('~/.cache/srtgen'), 'bin')
    name = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
    target = os.path.join(bindir, name)
    try:
        os.makedirs(bindir, exist_ok=True)
        reuse = (os.path.isfile(target) and not os.path.islink(target)
                 and os.path.getsize(target) == os.path.getsize(src))
        if not reuse:
            if os.path.islink(target) or os.path.exists(target):
                try:
                    os.remove(target)
                except OSError:
                    pass
            shutil.copy2(src, target)
            os.chmod(target, 0o755)
        _FFMPEG_PATH = target
        os.environ['PATH'] = bindir + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        # 退路：直接用 imageio 二进制目录（名字可能不符，但聊胜于无）
        _FFMPEG_PATH = src
        os.environ['PATH'] = os.path.dirname(src) + os.pathsep + os.environ.get('PATH', '')
    return _FFMPEG_PATH


def have_ffmpeg():
    return bool(_FFMPEG_PATH) or shutil.which('ffmpeg') is not None


# ----------------------------- 进度补丁（尽力而为） -----------------------------

class _ProgressReporter:
    """承载当前转录的进度回调（单转录串行执行，全局即可）。"""
    callback = None


class _TqdmShim:
    """替换 whisper/mlx_whisper.transcribe 内部 tqdm 的安全垫片。

    支持 `tqdm(...)` 与 `tqdm.tqdm(...)` 两种调用形态，并对未知属性返回空操作，
    确保即便上游内部结构有变化也不会破坏转录本身。
    """

    class _Bar:
        def __init__(self, iterable=None, total=None, **kw):
            self.iterable = iterable
            self.total = total
            self.n = 0

        def update(self, k=1):
            self.n += k
            cb = _ProgressReporter.callback
            if cb and self.total:
                try:
                    cb(max(0.0, min(1.0, self.n / float(self.total))))
                except Exception:
                    pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            for obj in (self.iterable or []):
                yield obj
                self.update(1)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    # 同时支持 shim(...) 与 shim.tqdm(...)
    tqdm = _Bar

    def __call__(self, *a, **k):
        return _TqdmShim._Bar(*a, **k)


def _install_progress_patch(module_name):
    """把指定 *.transcribe 模块的 tqdm 替换为垫片，返回还原函数。"""
    mod = sys.modules.get(module_name + '.transcribe')
    if mod is None or getattr(mod, 'tqdm', None) is None:
        return lambda: None
    original = mod.tqdm
    mod.tqdm = _TqdmShim()

    def restore():
        mod.tqdm = original

    return restore


# ----------------------------- 模型缓存 -----------------------------

_WHISPER_MODEL_CACHE = {}


def _get_whisper_model(whisper, size, device):
    key = (size, device)
    model = _WHISPER_MODEL_CACHE.get(key)
    if model is None:
        model = whisper.load_model(size, device=device)
        _WHISPER_MODEL_CACHE[key] = model
    return model


# ----------------------------- SRT 生成 -----------------------------

def format_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis = 0
        secs += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments):
    parts = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(float(segment['start']))
        end = format_timestamp(float(segment['end']))
        text = segment['text'].strip()
        parts.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(parts) + ("\n" if parts else "")


# ----------------------------- 转录线程 -----------------------------

class Worker(QThread):
    result = pyqtSignal(object)          # 最终结果（list 或错误 str）
    progress = pyqtSignal(str)           # 状态文本
    progress_pct = pyqtSignal(int)       # 0..100，转录进度（可能不触发）
    started_task = pyqtSignal(str)       # downloading / loading / transcribing

    def __init__(self, file_paths, model_size, device, language, task, export_itt, endpoint=None):
        super().__init__()
        self.file_paths = list(file_paths)
        self.model_size = model_size
        self.device = device
        self.language = language        # None 表示自动检测
        self.task = task                # 'transcribe' / 'translate'
        self.export_itt = export_itt
        self.endpoint = endpoint        # HF 下载端点（镜像）

    def _emit_pct(self, fraction):
        self.progress_pct.emit(int(fraction * 100))

    def _on_download_start(self):
        self.started_task.emit('downloading')

    def _on_download_progress(self, done, total, speed):
        """下载进度回调（节流到约 5 次/秒），显示百分比 + 速度 + 大小。"""
        now = time.time()
        if now - getattr(self, '_last_dl_emit', 0) < 0.2 and total and done < total:
            return
        self._last_dl_emit = now
        mb = 1 << 20
        pct = int(done / total * 100) if total else 0
        self.progress_pct.emit(pct)
        self.progress.emit(
            f'下载模型 {pct}% · {speed / mb:.1f} MB/s · {done // mb}/{(total or done) // mb} MB')

    def _transcribe_one(self, backend, path, apple, model_holder):
        """转录单个文件，返回 whisper 风格 result dict。"""
        if apple:
            # 多线程下载模型（带进度/速度），失败回退到传 repo id 让后端自行下载
            repo = model_mlx_repo(self.model_size)
            try:
                local = downloader.ensure_mlx_model(
                    repo,
                    on_progress=self._on_download_progress,
                    on_start=self._on_download_start,
                    endpoint=self.endpoint)
                if local:
                    repo = local
            except Exception:
                repo = model_mlx_repo(self.model_size)
            self.started_task.emit('transcribing')
            self.progress.emit('正在转录...')
            self.progress_pct.emit(0)
            restore = _install_progress_patch('mlx_whisper')
            _ProgressReporter.callback = self._emit_pct
            try:
                return backend.transcribe(
                    path,
                    path_or_hf_repo=repo,
                    language=self.language,
                    task=self.task,
                    verbose=False,  # 启用内部 tqdm，供进度垫片捕获
                )
            finally:
                _ProgressReporter.callback = None
                restore()
        else:
            if model_holder.get('model') is None:
                wname = model_whisper_name(self.model_size)
                try:
                    downloader.ensure_whisper_model(
                        wname,
                        on_progress=self._on_download_progress,
                        on_start=self._on_download_start)
                except Exception:
                    pass  # 回退到 whisper.load_model 自带下载
                self.started_task.emit('loading')
                self.progress.emit('正在加载模型...')
                model_holder['model'] = _get_whisper_model(backend, wname, self.device)
            self.started_task.emit('transcribing')
            self.progress.emit('正在转录...')
            self.progress_pct.emit(0)
            restore = _install_progress_patch('whisper')
            _ProgressReporter.callback = self._emit_pct
            try:
                return model_holder['model'].transcribe(
                    path,
                    language=self.language,
                    task=self.task,
                    verbose=False,
                )
            finally:
                _ProgressReporter.callback = None
                restore()

    def run(self):
        try:
            apple = is_apple_silicon()
            if apple:
                import mlx_whisper as backend
            else:
                import whisper as backend
        except Exception as e:
            self.result.emit(f'错误：加载转录引擎失败：{e}')
            return

        results = []
        model_holder = {'model': None}
        total = len(self.file_paths)

        for idx, path in enumerate(self.file_paths, 1):
            base = os.path.basename(path)
            if not os.path.exists(path):
                results.append((path, None, '文件不存在'))
                continue
            if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
                results.append((path, None, '不支持的文件格式'))
                continue

            if total > 1:
                self.progress.emit(f'处理中 {idx}/{total}：{base}')

            try:
                res = self._transcribe_one(backend, path, apple, model_holder)
                segments = res.get('segments') if isinstance(res, dict) else None
                if not segments:
                    raise ValueError('未能生成有效的字幕分段')

                srt_content = generate_srt(segments)
                srt_path = str(Path(path).with_suffix('.srt'))
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)

                itt_path = None
                if self.export_itt:
                    itt_path = str(Path(path).with_suffix('.itt'))
                    detected = res.get('language') if isinstance(res, dict) else None
                    srt2itt.convert_srt_to_itt(
                        srt_path, itt_path, lang=(self.language or detected or 'zh'))

                results.append((path, srt_path, None))
            except Exception as e:  # noqa: BLE001 - 逐文件汇总错误
                results.append((path, None, str(e)))

        self.result.emit(results)


# ----------------------------- 预下载线程 -----------------------------

class DownloadWorker(QThread):
    """仅下载所选模型（不转录），用于「预下载」。"""
    progress = pyqtSignal(str)
    progress_pct = pyqtSignal(int)
    done = pyqtSignal(str)  # '' 成功，否则错误信息

    def __init__(self, model_id, endpoint=None):
        super().__init__()
        self.model_id = model_id
        self.endpoint = endpoint
        self._last_emit = 0.0

    def _on_progress(self, done, total, speed):
        now = time.time()
        if now - self._last_emit < 0.2 and total and done < total:
            return
        self._last_emit = now
        mb = 1 << 20
        pct = int(done / total * 100) if total else 0
        self.progress_pct.emit(pct)
        self.progress.emit(
            f'下载中 {pct}% · {speed / mb:.1f} MB/s · {done // mb}/{(total or done) // mb} MB')

    def run(self):
        try:
            if is_apple_silicon():
                downloader.ensure_mlx_model(model_mlx_repo(self.model_id),
                                            on_progress=self._on_progress,
                                            endpoint=self.endpoint)
            else:
                if downloader.ensure_whisper_model(model_whisper_name(self.model_id),
                                                   on_progress=self._on_progress) is None:
                    raise RuntimeError('未知模型')
            self.done.emit('')
        except Exception as e:  # noqa: BLE001
            self.done.emit(str(e))


# ----------------------------- 主窗口 -----------------------------

STYLESHEET = """
QWidget { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
          font-size: 13px; color: #20232a; }
#central { background: #f5f6fb; }
#title { font-size: 17px; font-weight: 700; color: #4f46e5; }
#dropArea { background: #ffffff; border: 2px dashed #c5c8e0; border-radius: 14px; }
#dropArea[active="true"] { border-color: #7c3aed; background: #f4f1ff; }
#dropLabel { color: #6b7280; }
QLabel#fieldLabel { color: #4b5563; }
QComboBox { padding: 6px 10px; border: 1px solid #d6d9e6; border-radius: 9px; background: #ffffff; }
QComboBox:hover { border-color: #b3a8e8; }
QComboBox:disabled { background: #eef0f6; color: #9aa0ad; }
QPushButton { padding: 7px 12px; border: 1px solid #d6d9e6; border-radius: 9px; background: #ffffff; }
QPushButton:hover { background: #f0eefc; border-color: #b3a8e8; }
QPushButton:disabled { color: #b8bcc8; background: #f2f3f8; }
QPushButton#primary { background: #6d28d9; color: #ffffff; border: none; font-weight: 600;
                      font-size: 14px; padding: 11px; }
QPushButton#primary:hover { background: #7c3aed; }
QPushButton#primary:disabled { background: #c7bbf2; color: #ffffff; }
QPushButton#ghost { border: none; color: #4f46e5; background: transparent; }
QPushButton#ghost:hover { color: #7c3aed; background: transparent; }
QPushButton#danger:enabled { color: #b91c1c; }
QProgressBar { border: none; border-radius: 7px; background: #e7e8f2; height: 14px;
               text-align: center; color: #20232a; }
QProgressBar::chunk { background: #7c3aed; border-radius: 7px; }
#cacheInfo { color: #6b7280; font-size: 12px; }
#status { color: #4b5563; }
#elapsed { color: #9aa0ad; font-size: 12px; }
QCheckBox { spacing: 6px; }
"""


class DropArea(QFrame):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('dropArea')
        self.setMinimumHeight(120)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit()


def _fmt_size(n):
    mb = n / (1 << 20)
    return f'{mb / 1024:.1f} GB' if mb >= 1024 else f'{mb:.0f} MB'


class SubtitleGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Whisper 字幕生成器')
        self.setMinimumWidth(460)
        self.file_paths = []
        self.start_time = None
        self.current_task = None
        self.worker = None
        self.dl_worker = None
        self._busy = False
        self.initUI()
        self.update_cache_status()

    def _field_row(self, label_text, widget):
        row = QHBoxLayout()
        lab = QLabel(label_text)
        lab.setObjectName('fieldLabel')
        lab.setFixedWidth(48)
        row.addWidget(lab)
        row.addWidget(widget, 1)
        return row

    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QLabel('Whisper 字幕生成器')
        title.setObjectName('title')
        layout.addWidget(title)

        # 拖拽区（可点击）
        self.drop_area = DropArea()
        self.drop_area.clicked.connect(self.open_file_dialog)
        da_layout = QVBoxLayout(self.drop_area)
        da_layout.setContentsMargins(16, 16, 16, 16)
        self.label = QLabel('拖拽音视频文件到此处\n或点击选择（支持多选）')
        self.label.setObjectName('dropLabel')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        da_layout.addWidget(self.label)
        layout.addWidget(self.drop_area)

        # 模型选择
        self.model_selector = QComboBox(self)
        for model_id, desc, *_ in MODELS:
            self.model_selector.addItem(desc, model_id)
        self.model_selector.currentIndexChanged.connect(self.update_cache_status)
        layout.addLayout(self._field_row('模型', self.model_selector))

        # 模型缓存状态 + 预下载/删除
        cache_row = QHBoxLayout()
        cache_row.addSpacing(56)
        self.cache_info = QLabel('')
        self.cache_info.setObjectName('cacheInfo')
        cache_row.addWidget(self.cache_info, 1)
        self.predownload_btn = QPushButton('预下载')
        self.predownload_btn.clicked.connect(self.predownload_model)
        self.delete_btn = QPushButton('删除缓存')
        self.delete_btn.setObjectName('danger')
        self.delete_btn.clicked.connect(self.delete_model_cache)
        cache_row.addWidget(self.predownload_btn)
        cache_row.addWidget(self.delete_btn)
        layout.addLayout(cache_row)

        # 语言
        self.language_selector = QComboBox(self)
        for code, name in LANGUAGES:
            self.language_selector.addItem(name, code)
        self.language_selector.setCurrentIndex(1)  # 默认中文
        layout.addLayout(self._field_row('语言', self.language_selector))

        # 任务
        self.task_selector = QComboBox(self)
        self.task_selector.addItem('转录（保留原语言）', 'transcribe')
        self.task_selector.addItem('翻译成英文', 'translate')
        layout.addLayout(self._field_row('任务', self.task_selector))

        # 设备
        self.device_selector = QComboBox(self)
        if not is_apple_silicon():
            if self.check_cuda():
                self.device_selector.addItem('CUDA（GPU 加速）', 'cuda')
            self.device_selector.addItem('CPU', 'cpu')
        else:
            self.device_selector.addItem('MLX（Apple Silicon）', 'mlx')
            self.device_selector.setEnabled(False)
        layout.addLayout(self._field_row('设备', self.device_selector))

        # 下载源（镜像可大幅提升国内下载速度；只影响 Apple Silicon 的模型下载）
        self.source_selector = QComboBox(self)
        self.source_selector.addItem('国内镜像 hf-mirror.com（推荐）', 'https://hf-mirror.com')
        self.source_selector.addItem('官方 HuggingFace', None)
        layout.addLayout(self._field_row('下载源', self.source_selector))

        self.itt_checkbox = QCheckBox('同时导出 Apple .itt 字幕', self)
        layout.addWidget(self.itt_checkbox)

        self.generate_button = QPushButton('生成字幕', self)
        self.generate_button.setObjectName('primary')
        self.generate_button.clicked.connect(self.generate_subtitle)
        layout.addWidget(self.generate_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel('')
        self.status_label.setObjectName('status')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.time_label = QLabel('')
        self.time_label.setObjectName('elapsed')
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setVisible(False)
        layout.addWidget(self.time_label)

        layout.addStretch(1)

        container = QWidget()
        container.setObjectName('central')
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setAcceptDrops(True)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.setInterval(1000)

    # --- 模型缓存管理 ---
    def update_cache_status(self):
        mid = self.model_selector.currentData()
        if not mid:
            return
        try:
            cached, size = downloader.model_cache_info(
                is_apple_silicon(), model_mlx_repo(mid), model_whisper_name(mid))
        except Exception:
            cached, size = False, 0
        if cached:
            self.cache_info.setText(f'✓ 已缓存 · {_fmt_size(size)}')
        else:
            self.cache_info.setText(f'未下载 · 约 {model_approx_mb(mid)} MB')
        self.predownload_btn.setEnabled(not cached and not self._busy)
        self.delete_btn.setEnabled(cached and not self._busy)

    def predownload_model(self):
        if not have_ffmpeg():
            pass  # 预下载不需要 ffmpeg
        self._set_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText('准备下载...')
        self.current_task = 'downloading'
        self.start_timer()
        self.dl_worker = DownloadWorker(self.model_selector.currentData(),
                                        self.source_selector.currentData())
        self.dl_worker.progress.connect(self.update_progress)
        self.dl_worker.progress_pct.connect(self.update_pct)
        self.dl_worker.done.connect(self.on_predownload_done)
        self.dl_worker.start()

    def on_predownload_done(self, err):
        self.progress_bar.setVisible(False)
        self.stop_timer()
        self._set_busy(False)
        if err:
            self.status_label.setText(f'下载失败：{err}')
        else:
            self.status_label.setText('模型已下载完成 ✓')

    def delete_model_cache(self):
        mid = self.model_selector.currentData()
        reply = QMessageBox.question(
            self, '删除缓存',
            f'确定删除「{_MODEL_BY_ID[mid][1]}」的本地缓存吗？\n（之后再次使用会重新下载）',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            freed = downloader.delete_model_cache(
                is_apple_silicon(), model_mlx_repo(mid), model_whisper_name(mid))
            self.status_label.setText(f'已删除缓存，释放 {_fmt_size(freed)}')
        except Exception as e:
            self.status_label.setText(f'删除失败：{e}')
        self.update_cache_status()

    def _set_busy(self, busy):
        self._busy = busy
        self.generate_button.setDisabled(busy)
        self.model_selector.setDisabled(busy)
        self.drop_area.setDisabled(busy)
        if busy:
            self.predownload_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
        else:
            self.update_cache_status()

    # --- 计时 ---
    def start_timer(self):
        self.start_time = time.time()
        self.timer.start()
        self.time_label.setVisible(True)

    def stop_timer(self):
        self.timer.stop()
        self.time_label.setVisible(False)
        self.start_time = None
        self.current_task = None

    def update_time(self):
        if self.start_time is None:
            return
        elapsed = int(time.time() - self.start_time)
        minutes, seconds = divmod(elapsed, 60)
        self.time_label.setText(f'用时 {minutes}分{seconds}秒')

    def check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # --- 文件选择 / 拖拽 ---
    def _set_drop_active(self, on):
        self.drop_area.setProperty('active', 'true' if on else 'false')
        self.drop_area.style().unpolish(self.drop_area)
        self.drop_area.style().polish(self.drop_area)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and not self._busy:
            event.accept()
            self._set_drop_active(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_drop_active(False)

    def dropEvent(self, event):
        self._set_drop_active(False)
        all_paths = [u.toLocalFile() for u in event.mimeData().urls()]
        supported = [p for p in all_paths if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS]
        if supported:
            self.set_files(supported)
        elif all_paths:
            self.label.setText('格式不支持，请拖入音频/视频文件')

    def open_file_dialog(self):
        if self._busy:
            return
        audio_filter = ' '.join(f'*{ext}' for ext in sorted(SUPPORTED_AUDIO_EXTENSIONS))
        video_filter = ' '.join(f'*{ext}' for ext in sorted(SUPPORTED_VIDEO_EXTENSIONS))
        names, _ = QFileDialog.getOpenFileNames(
            self, '选择文件', '',
            f'支持的文件 ({audio_filter} {video_filter});;'
            f'音频文件 ({audio_filter});;视频文件 ({video_filter});;所有文件 (*.*)',
        )
        if names:
            self.set_files(names)

    def set_files(self, paths):
        self.file_paths = paths
        if len(paths) == 1:
            self.label.setText(f'已选择：{os.path.basename(paths[0])}')
        else:
            self.label.setText(f'已选择 {len(paths)} 个文件')

    # --- 生成 ---
    def generate_subtitle(self):
        if not self.file_paths:
            self.status_label.setText('请先选择至少一个文件')
            return
        if not have_ffmpeg():
            self.status_label.setText('错误：未找到 ffmpeg，请重新安装应用或在系统中安装 ffmpeg。')
            return

        self._set_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 先用忙碌态，收到进度后转确定态

        self.worker = Worker(
            self.file_paths,
            self.model_selector.currentData(),
            self.device_selector.currentData(),
            self.language_selector.currentData(),
            self.task_selector.currentData(),
            self.itt_checkbox.isChecked(),
            self.source_selector.currentData(),
        )
        self.worker.result.connect(self.on_result)
        self.worker.progress.connect(self.update_progress)
        self.worker.progress_pct.connect(self.update_pct)
        self.worker.started_task.connect(self.on_task_started)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self.status_label.setText('初始化中...')

    def on_task_started(self, task):
        self.current_task = task
        if task == 'transcribing':
            # 转录开始：保持忙碌态，等首个百分比再切换为确定态
            self.progress_bar.setRange(0, 0)
        if self.start_time is None:
            self.start_timer()

    def update_progress(self, message):
        if not message.startswith('错误'):
            self.status_label.setText(message)

    def update_pct(self, pct):
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(pct)

    def on_result(self, payload):
        if isinstance(payload, str):  # 引擎级错误
            self.status_label.setText(payload)
            return
        ok = [r for r in payload if r[2] is None]
        failed = [r for r in payload if r[2] is not None]
        if len(payload) == 1 and ok:
            self.status_label.setText(f'字幕已生成：{os.path.basename(ok[0][1])}')
        else:
            msg = f'完成：成功 {len(ok)} 个，失败 {len(failed)} 个'
            if failed:
                first = failed[0]
                msg += f'（{os.path.basename(first[0])}：{first[2]}）'
            self.status_label.setText(msg)

    def on_worker_finished(self):
        self.progress_bar.setVisible(False)
        self.stop_timer()
        self._set_busy(False)


def _check_backend_assets():
    """检查所选后端的随包数据资源是否齐全；返回缺失列表。"""
    import importlib
    name = 'mlx_whisper' if is_apple_silicon() else 'whisper'
    mod = importlib.import_module(name)
    base = os.path.join(os.path.dirname(mod.__file__), 'assets')
    needed = ['mel_filters.npz', 'multilingual.tiktoken', 'gpt2.tiktoken']
    return [f for f in needed if not os.path.exists(os.path.join(base, f))]


def selftest():
    """冒烟测试：验证冻结产物里 ffmpeg、后端资源、tiktoken、Qt 插件齐全。

    用于 CI 在打包后直接运行 `SRT_gen --selftest`，无需下载模型即可发现
    “编译通过但运行时崩溃”的缺资源问题。成功返回 0，失败返回非 0。
    """
    import importlib
    ok = True
    print(f'[selftest] platform={platform.platform()} machine={platform.machine()} frozen={getattr(sys, "frozen", False)}')

    # 逐个探测重型依赖，便于一次构建即发现所有打包缺失
    probe = ['numpy', 'numba'] + (['scipy', 'mlx.core'] if is_apple_silicon() else ['torch'])
    for _m in probe:
        try:
            importlib.import_module(_m)
            print(f'[selftest] import {_m} OK')
        except Exception as e:
            ok = False
            print(f'[selftest] FAIL: import {_m}: {e!r}')

    ff = setup_ffmpeg()
    print(f'[selftest] ffmpeg={ff} have={have_ffmpeg()}')
    if not have_ffmpeg():
        ok = False
        print('[selftest] FAIL: ffmpeg 不可用')
    # 关键：以子进程方式（依赖 PATH）调用 ffmpeg —— whisper/mlx_whisper 正是这样用的
    try:
        import subprocess
        r = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and 'ffmpeg version' in (r.stdout + r.stderr):
            print('[selftest] ffmpeg 子进程(PATH)可执行 OK')
        else:
            ok = False
            print(f'[selftest] FAIL: ffmpeg 子进程返回码 {r.returncode}')
    except Exception as e:
        ok = False
        print(f'[selftest] FAIL: 无法以子进程方式调用 ffmpeg: {e!r}')

    try:
        missing = _check_backend_assets()
        if missing:
            ok = False
            print(f'[selftest] FAIL: 后端缺少资源 {missing}')
        else:
            print('[selftest] 后端数据资源 OK')
    except Exception as e:
        ok = False
        print(f'[selftest] FAIL: 导入后端失败 {e!r}')

    # tiktoken / tokenizer（验证 tiktoken_ext 隐藏导入与 *.tiktoken 资源）
    try:
        if is_apple_silicon():
            from mlx_whisper.tokenizer import get_tokenizer
        else:
            from whisper.tokenizer import get_tokenizer
        get_tokenizer(multilingual=True)
        print('[selftest] tokenizer OK')
    except Exception as e:
        ok = False
        print(f'[selftest] FAIL: tokenizer 失败 {e!r}')

    # 下载器模块（确保随包）
    try:
        import downloader as _dl
        assert hasattr(_dl, 'parallel_download')
        print('[selftest] downloader OK')
    except Exception as e:
        ok = False
        print(f'[selftest] FAIL: downloader {e!r}')

    # MLX Metal 内核：强制一次 GPU 计算以加载 mlx.metallib（验证其已打包）
    if is_apple_silicon():
        try:
            import mlx.core as mx
            res = mx.ones((4, 4)) + mx.ones((4, 4))
            mx.eval(res)
            assert float(res.sum()) == 32.0
            print('[selftest] mlx Metal 内核 OK')
        except Exception as e:
            ok = False
            print(f'[selftest] FAIL: mlx 计算失败（metallib 缺失?） {e!r}')

    # Qt 平台插件（offscreen 验证 PyQt5 插件已打包）
    try:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        _app = QApplication.instance() or QApplication(sys.argv[:1])
        icon_path = resource_path(os.path.join('assets', 'icon.png'))
        print(f'[selftest] Qt OK; icon_exists={os.path.exists(icon_path)}')
    except Exception as e:
        ok = False
        print(f'[selftest] FAIL: Qt 初始化失败 {e!r}')

    print(f'[selftest] {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


def cli_transcribe(path, model_id='tiny'):
    """命令行转录单个文件（复用 Worker，验证真实流程；用于测试 GUI 启动环境下 ffmpeg 是否可用）。"""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    setup_ffmpeg()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    holder = {}
    device = 'mlx' if is_apple_silicon() else 'cpu'
    w = Worker([path], model_id, device, None, 'transcribe', False)
    w.result.connect(lambda r: holder.update(r=r))
    w.progress.connect(lambda m: print('[progress]', m, flush=True))
    w.finished.connect(app.quit)
    w.start()
    app.exec_()
    r = holder.get('r')
    if isinstance(r, list) and r and r[0][2] is None:
        print('[OK] SRT:', r[0][1])
        return 0
    print('[FAIL]', r)
    return 1


def main():
    setup_ffmpeg()
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    icon_path = resource_path(os.path.join('assets', 'icon.png'))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = SubtitleGenerator()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(selftest())
    if '--transcribe' in sys.argv:
        _i = sys.argv.index('--transcribe')
        sys.exit(cli_transcribe(sys.argv[_i + 1]))
    main()
