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
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QPushButton,
    QWidget, QComboBox, QProgressBar, QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon

import srt2itt

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

MODELS = [
    ('large-v3', '大型 V3（最新，准确度最高）'),
    ('large-v2', '大型 V2'),
    ('large', '大型'),
    ('medium', '中型'),
    ('small', '小型'),
    ('base', '基础'),
    ('tiny', '最小'),
]


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
    mlx_whisper 内部以 `ffmpeg` 调用子进程，故创建一个名为 ffmpeg 的软链/副本并
    把其目录前置到 PATH。若随包二进制不可用，则回退到系统 PATH 中的 ffmpeg。
    """
    global _FFMPEG_PATH
    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _FFMPEG_PATH = shutil.which('ffmpeg')
        return _FFMPEG_PATH

    bindir = os.path.join(tempfile.gettempdir(), 'srtgen_bin')
    name = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
    target = os.path.join(bindir, name)
    try:
        os.makedirs(bindir, exist_ok=True)
        if not os.path.exists(target):
            try:
                os.symlink(src, target)
            except (OSError, NotImplementedError, AttributeError):
                shutil.copy2(src, target)
                os.chmod(target, 0o755)
        _FFMPEG_PATH = target
    except Exception:
        _FFMPEG_PATH = src  # 至少本程序可直接用该路径

    os.environ['PATH'] = os.path.dirname(_FFMPEG_PATH) + os.pathsep + os.environ.get('PATH', '')
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


def _whisper_pt_cached(size):
    return os.path.exists(os.path.expanduser(f'~/.cache/whisper/{size}.pt'))


def _mlx_repo_cached(size):
    repo = f'models--mlx-community--whisper-{size}-mlx'
    hub = os.path.expanduser('~/.cache/huggingface/hub')
    return os.path.isdir(os.path.join(hub, repo))


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

    def __init__(self, file_paths, model_size, device, language, task, export_itt):
        super().__init__()
        self.file_paths = list(file_paths)
        self.model_size = model_size
        self.device = device
        self.language = language        # None 表示自动检测
        self.task = task                # 'transcribe' / 'translate'
        self.export_itt = export_itt

    def _emit_pct(self, fraction):
        self.progress_pct.emit(int(fraction * 100))

    def _transcribe_one(self, backend, path, apple, model_holder):
        """转录单个文件，返回 whisper 风格 result dict。"""
        if apple:
            repo = f'mlx-community/whisper-{self.model_size}-mlx'
            if not _mlx_repo_cached(self.model_size):
                self.started_task.emit('downloading')
                self.progress.emit(f'首次使用，正在下载 {self.model_size} 模型并转录（较慢）...')
            else:
                self.started_task.emit('transcribing')
                self.progress.emit('正在转录...')
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
                if _whisper_pt_cached(self.model_size):
                    self.started_task.emit('loading')
                    self.progress.emit('正在加载模型...')
                else:
                    self.started_task.emit('downloading')
                    self.progress.emit(f'正在下载 {self.model_size} 模型（首次使用较慢）...')
                model_holder['model'] = _get_whisper_model(backend, self.model_size, self.device)
            self.started_task.emit('transcribing')
            self.progress.emit('正在转录...')
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


# ----------------------------- 主窗口 -----------------------------

class SubtitleGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Whisper 字幕生成器')
        self.setGeometry(200, 200, 460, 460)
        self.file_paths = []
        self.start_time = None
        self.current_task = None
        self.worker = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        self.label = QLabel('拖拽音视频文件到此处，或点击下方按钮选择（支持多选）', self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.time_label = QLabel('', self)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setVisible(False)
        layout.addWidget(self.time_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.button = QPushButton('选择文件', self)
        self.button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button)

        layout.addWidget(QLabel('模型大小（越大越准但越慢）：'))
        self.model_selector = QComboBox(self)
        for model_id, desc in MODELS:
            self.model_selector.addItem(desc, model_id)
        layout.addWidget(self.model_selector)

        layout.addWidget(QLabel('语言：'))
        self.language_selector = QComboBox(self)
        for code, name in LANGUAGES:
            self.language_selector.addItem(name, code)
        self.language_selector.setCurrentIndex(1)  # 默认中文
        layout.addWidget(self.language_selector)

        layout.addWidget(QLabel('任务：'))
        self.task_selector = QComboBox(self)
        self.task_selector.addItem('转录（保留原语言）', 'transcribe')
        self.task_selector.addItem('翻译成英文', 'translate')
        layout.addWidget(self.task_selector)

        layout.addWidget(QLabel('运行设备：'))
        self.device_selector = QComboBox(self)
        if not is_apple_silicon():
            if self.check_cuda():
                self.device_selector.addItem('CUDA（GPU 加速）', 'cuda')
            self.device_selector.addItem('CPU', 'cpu')
        else:
            self.device_selector.addItem('MLX（Apple Silicon）', 'mlx')
            self.device_selector.setEnabled(False)
        layout.addWidget(self.device_selector)

        self.itt_checkbox = QCheckBox('同时导出 Apple .itt 字幕', self)
        layout.addWidget(self.itt_checkbox)

        self.generate_button = QPushButton('生成字幕', self)
        self.generate_button.clicked.connect(self.generate_subtitle)
        layout.addWidget(self.generate_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setAcceptDrops(True)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.setInterval(1000)

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
        messages = {
            'downloading': f'正在下载模型... ({minutes}分{seconds}秒)',
            'loading': f'正在加载模型... ({minutes}分{seconds}秒)',
            'transcribing': f'正在转录... ({minutes}分{seconds}秒)',
        }
        if self.current_task in messages:
            self.time_label.setText(messages[self.current_task])

    def check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # --- 文件选择 / 拖拽 ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        paths = [p for p in paths if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS]
        if paths:
            self.set_files(paths)

    def open_file_dialog(self):
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
            self.label.setText(f'已选择文件: {os.path.basename(paths[0])}')
        else:
            self.label.setText(f'已选择 {len(paths)} 个文件')

    # --- 生成 ---
    def generate_subtitle(self):
        if not self.file_paths:
            self.label.setText('请先选择至少一个文件')
            return
        if not have_ffmpeg():
            self.label.setText('错误：未找到 ffmpeg，请重新安装应用或在系统中安装 ffmpeg。')
            return

        self.generate_button.setEnabled(False)
        self.button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 先用忙碌态，收到进度后转确定态

        self.worker = Worker(
            self.file_paths,
            self.model_selector.currentData(),
            self.device_selector.currentData(),
            self.language_selector.currentData(),
            self.task_selector.currentData(),
            self.itt_checkbox.isChecked(),
        )
        self.worker.result.connect(self.on_result)
        self.worker.progress.connect(self.update_progress)
        self.worker.progress_pct.connect(self.update_pct)
        self.worker.started_task.connect(self.on_task_started)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self.label.setText('初始化中...')

    def on_task_started(self, task):
        self.current_task = task
        if task == 'transcribing':
            # 转录开始：保持忙碌态，等首个百分比再切换为确定态
            self.progress_bar.setRange(0, 0)
        if self.start_time is None:
            self.start_timer()

    def update_progress(self, message):
        if not message.startswith('错误'):
            self.label.setText(message)

    def update_pct(self, pct):
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(pct)

    def on_result(self, payload):
        if isinstance(payload, str):  # 引擎级错误
            self.label.setText(payload)
            return
        ok = [r for r in payload if r[2] is None]
        failed = [r for r in payload if r[2] is not None]
        if len(payload) == 1 and ok:
            self.label.setText(f'字幕已生成: {os.path.basename(ok[0][1])}')
        else:
            msg = f'完成：成功 {len(ok)} 个，失败 {len(failed)} 个'
            if failed:
                first = failed[0]
                msg += f'\n（{os.path.basename(first[0])}：{first[2]}）'
            self.label.setText(msg)

    def on_worker_finished(self):
        self.generate_button.setEnabled(True)
        self.button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.stop_timer()


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


def main():
    setup_ffmpeg()
    app = QApplication(sys.argv)
    icon_path = resource_path(os.path.join('assets', 'icon.png'))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = SubtitleGenerator()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(selftest())
    main()
