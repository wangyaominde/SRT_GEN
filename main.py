import sys
import whisper
import os
import shutil
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QPushButton, QWidget, QComboBox, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from pathlib import Path
import time
import subprocess
import json

class Worker(QThread):
    result = pyqtSignal(object)
    progress = pyqtSignal(str)
    started_task = pyqtSignal(str)

    def __init__(self, file_path, model_size, device):
        super().__init__()
        self.file_path = file_path
        self.model_size = model_size
        self.device = device

    def get_audio_duration(self):
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                  '-of', 'json', self.file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            return float(data['format']['duration'])
        except Exception as e:
            raise ValueError(f"无法获取文件时长: {str(e)}")

    def run(self):
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError("所选文件不存在")

            valid_extensions = {'.mp3', '.wav', '.mp4', '.mkv', '.mov'}
            if not Path(self.file_path).suffix.lower() in valid_extensions:
                raise ValueError("不支持的文件格式")

            # 检查文件时长
            duration = self.get_audio_duration()
            if duration < 10:
                raise ValueError(f"文件时长太短（{duration:.1f}秒），最少需要10秒")

            # 加载模型
            model_path = os.path.expanduser(f"~/.cache/whisper/{self.model_size}.pt")
            if not os.path.exists(model_path):
                self.started_task.emit("downloading")
                self.progress.emit(f"正在下载 {self.model_size} 模型，首次使用需要较长时间...")
            else:
                self.started_task.emit("loading")
                self.progress.emit("正在加载模型...")
            
            model = whisper.load_model(self.model_size, device=self.device)
            
            self.started_task.emit("transcribing")
            self.progress.emit("正在转录音频，请耐心等待...")

            result = model.transcribe(
                self.file_path,
                language='zh',
                verbose=False
            )
            
            if not result or 'segments' not in result:
                raise ValueError("转录失败，未能生成有效的分段数据")
                
            self.result.emit(result['segments'])
            
        except FileNotFoundError as e:
            self.result.emit(f"错误：{str(e)}")
        except ValueError as e:
            self.result.emit(f"错误：{str(e)}")
        except Exception as e:
            self.result.emit(f"发生错误：{str(e)}")

class SubtitleGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Whisper 字幕生成器')
        self.setGeometry(200, 200, 400, 300)  # 缩小窗口尺寸
        self.file_path = ""
        self.start_time = None
        self.current_task = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)  # 设置间距

        # 状态标签
        self.label = QLabel('拖拽音视频文件到此处，或点击按钮选择文件\n(文件时长需要大于10秒)', self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # 时间标签
        self.time_label = QLabel('', self)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setVisible(False)
        layout.addWidget(self.time_label)

        # 进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 文件选择按钮
        self.button = QPushButton('选择文件', self)
        self.button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button)

        # 模型选择
        model_label = QLabel('选择模型大小（越大准确度越高，但速度更慢）：')
        layout.addWidget(model_label)
        
        self.model_selector = QComboBox(self)
        models = [
            ("tiny", "最小 (速度最快，准确度最低)"),
            ("base", "基础 (较快，准确度一般)"),
            ("small", "小型 (平衡速度和准确度)"),
            ("medium", "中型 (较慢，准确度较高)"),
            ("large", "大型 (最慢，准确度最高)")
        ]
        for model_id, model_desc in models:
            self.model_selector.addItem(model_desc, model_id)
        layout.addWidget(self.model_selector)

        # 设备选择
        device_label = QLabel('选择运行设备：')
        layout.addWidget(device_label)
        
        self.device_selector = QComboBox(self)
        if self.check_cuda():
            self.device_selector.addItem("CUDA (GPU加速)", "cuda")
        self.device_selector.addItem("CPU", "cpu")
        layout.addWidget(self.device_selector)

        # 生成按钮
        self.generate_button = QPushButton('生成字幕', self)
        self.generate_button.clicked.connect(self.generate_subtitle)
        layout.addWidget(self.generate_button)

        # 设置主容器
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setAcceptDrops(True)

        # 设置计时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.setInterval(1000)

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
        minutes = elapsed // 60
        seconds = elapsed % 60
        
        task_messages = {
            "downloading": f"正在下载模型... ({minutes}分{seconds}秒)",
            "loading": f"正在加载模型... ({minutes}分{seconds}秒)",
            "transcribing": f"正在转录音频... ({minutes}分{seconds}秒)"
        }
        
        if self.current_task in task_messages:
            self.label.setText(task_messages[self.current_task])

    def check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.file_path = urls[0].toLocalFile()
            self.label.setText(f'已选择文件: {os.path.basename(self.file_path)}')

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            '选择文件',
            '',
            '支持的文件 (*.mp3 *.wav *.mp4 *.mkv);;所有文件 (*.*)',
            options=options
        )
        if file_name:
            self.file_path = file_name
            self.label.setText(f'已选择文件: {os.path.basename(self.file_path)}')

    def generate_subtitle(self):
        if not self.file_path:
            self.label.setText('请先选择一个文件')
            return

        if not self.check_ffmpeg():
            self.label.setText('错误：未找到ffmpeg，请确保已安装ffmpeg并配置环境变量。')
            return

        self.generate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        model_size = self.model_selector.currentData()
        device = self.device_selector.currentData()

        self.worker = Worker(self.file_path, model_size, device)
        self.worker.result.connect(self.update_status)
        self.worker.progress.connect(self.update_progress)
        self.worker.started_task.connect(self.on_task_started)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        
        self.label.setText('初始化中...')

    def on_task_started(self, task):
        self.current_task = task
        self.start_timer()

    def check_ffmpeg(self):
        return shutil.which("ffmpeg") is not None

    def update_progress(self, message):
        if not message.startswith("错误"):
            self.label.setText(message)

    def update_status(self, message):
        if isinstance(message, str) and message.startswith("错误"):
            self.label.setText(message)
            self.progress_bar.setVisible(False)
            self.stop_timer()
        else:
            self.display_result(message)

    def on_worker_finished(self):
        self.generate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.stop_timer()

    def display_result(self, segments):
        if isinstance(segments, str):
            self.label.setText(segments)
            return

        try:
            srt_content = self.generate_srt(segments)
            srt_path = str(Path(self.file_path).with_suffix('.srt'))
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            self.label.setText(f'字幕已成功生成: {os.path.basename(srt_path)}')
            
        except Exception as e:
            self.label.setText(f'保存字幕文件时发生错误：{str(e)}')

    def generate_srt(self, segments):
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            start = self.format_timestamp(float(segment['start']))
            end = self.format_timestamp(float(segment['end']))
            text = segment['text'].strip()
            srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
        return srt_content

    def format_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SubtitleGenerator()
    window.show()
    sys.exit(app.exec_())