import sys
import whisper
import os
import shutil
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QPushButton, QWidget, QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import requests

class Worker(QThread):
    result = pyqtSignal(object)

    def __init__(self, file_path, model_size, device):
        super().__init__()
        self.file_path = file_path
        self.model_size = model_size
        self.device = device

    def run(self):
        try:
            model_url = f"https://huggingface.co/whisper/models/{self.model_size}.tar.gz"
            model_path = f"./models/{self.model_size}.tar.gz"
            model_dir = "./models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not os.path.exists(model_path):
                with requests.get(model_url, stream=True) as r:
                    with open(model_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=4096):
                            if chunk:
                                f.write(chunk)
            model = whisper.load_model(self.model_size, device=self.device)
            result = model.transcribe(self.file_path)
            self.result.emit(result['segments'])
        except FileNotFoundError:
            self.result.emit("错误：未找到ffmpeg，请确保已安装ffmpeg并配置环境变量。")
        except Exception as e:
            self.result.emit(f"发生错误：{str(e)}")

class SubtitleGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Whisper Subtitle Generator')
        self.setGeometry(200, 200, 600, 400)
        self.file_path = ""
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel('拖拽文件到此处，或点击按钮选择文件', self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.button = QPushButton('选择文件', self)
        self.button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(["tiny", "base", "small", "medium", "large"])
        layout.addWidget(self.model_selector)

        self.device_selector = QComboBox(self)
        self.device_selector.addItems(["cuda", "cpu"])
        layout.addWidget(self.device_selector)

        self.generate_button = QPushButton('生成字幕', self)
        self.generate_button.clicked.connect(self.generate_subtitle)
        layout.addWidget(self.generate_button)

        self.delete_model_button = QPushButton('删除不用的模型', self)
        self.delete_model_button.clicked.connect(self.delete_unused_models)
        layout.addWidget(self.delete_model_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 支持拖拽
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.file_path = urls[0].toLocalFile()
            self.label.setText(f'已选择文件: {self.file_path}')

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, '选择文件', '', '音频文件 (*.mp3 *.wav);;视频文件 (*.mp4 *.mkv)', options=options)
        if file_name:
            self.file_path = file_name
            self.label.setText(f'已选择文件: {self.file_path}')

    def generate_subtitle(self):
        if not self.file_path:
            self.label.setText('请先选择一个文件')
            return

        # 检查是否安装了 ffmpeg
        if not self.check_ffmpeg():
            self.label.setText('错误：未找到ffmpeg，请确保已安装ffmpeg并配置环境变量。')
            return

        model_size = self.model_selector.currentText()
        device = self.device_selector.currentText()

        # 使用多线程生成字幕
        self.worker = Worker(self.file_path, model_size, device)
        self.worker.result.connect(self.update_status)
        self.worker.start()
        self.label.setText('正在生成字幕...')

    def check_ffmpeg(self):
        # 检查系统路径中是否存在 ffmpeg
        ffmpeg_exists = shutil.which("ffmpeg")
        return ffmpeg_exists is not None

    def update_status(self, message):
        if isinstance(message, str) and (message.startswith("错误") or message.startswith("发生错误")):
            self.label.setText(message)
        else:
            self.display_result(message)

    def display_result(self, segments):
        if isinstance(segments, str) and (segments.startswith("错误") or segments.startswith("发生错误")):
            self.label.setText(segments)
        else:
            # 将转录文本保存为 SRT 格式
            srt_content = self.generate_srt(segments)
            srt_path = self.file_path.rsplit('.', 1)[0] + '.srt'
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            self.label.setText(f'字幕已生成: {srt_path}')

    def generate_srt(self, segments):
        # 使用转录结果中的每个片段来生成 SRT
        srt_content = ""
        for i, segment in enumerate(segments):
            start = self.format_timestamp(segment['start'])
            end = self.format_timestamp(segment['end'])
            srt_content += f"{i+1}\n{start} --> {end}\n{segment['text'].strip()}\n\n"
        return srt_content

    def format_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def delete_unused_models(self):
        model_dir = './models'
        if os.path.exists(model_dir):
            try:
                for file_name in os.listdir(model_dir):
                    file_path = os.path.join(model_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                self.label.setText('未使用的模型文件已删除')
            except Exception as e:
                self.label.setText(f'删除模型时发生错误：{str(e)}')
        else:
            self.label.setText('未找到模型目录，无需删除')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SubtitleGenerator()
    window.show()
    sys.exit(app.exec_())