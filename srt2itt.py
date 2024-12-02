import os
import re
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
import threading

def parse_srt_time(srt_time):
    """Converts SRT time format (hh:mm:ss,ms) to ITT time format (hh:mm:ss.ms)."""
    return srt_time.replace(',', '.')

def convert_srt_to_itt(srt_file, itt_file):
    """Converts a SRT subtitle file to an ITT subtitle file."""
    with open(srt_file, 'r', encoding='utf-8') as file:
        srt_content = file.read()

    # Regular expression to parse SRT subtitles
    srt_pattern = re.compile(r'^(\d+)\s+([\d:,]+) --> ([\d:,]+)\s+((?:.*(?:\n|$))+)\n?', re.MULTILINE)

    matches = srt_pattern.findall(srt_content)

    # Create the root element for ITT
    tt = ET.Element('tt', attrib={
        'xmlns': "http://www.w3.org/ns/ttml",
        'xmlns:tts': "http://www.w3.org/ns/ttml#styling",
        'xmlns:ttm': "http://www.w3.org/ns/ttml#metadata"
    })

    # Create body and div elements
    body = ET.SubElement(tt, 'body')
    div = ET.SubElement(body, 'div')

    # Convert each SRT entry to ITT format
    for match in matches:
        seq_num, start_time, end_time, text = match
        p = ET.SubElement(div, 'p', attrib={
            'begin': parse_srt_time(start_time),
            'end': parse_srt_time(end_time)
        })
        # Clean up the subtitle text and add to <p>
        p.text = text.replace('\n', ' ')

    # Write the ITT file
    tree = ET.ElementTree(tt)
    ET.indent(tree, space="  ", level=0)  # Pretty-print the XML (Python 3.9+)
    tree.write(itt_file, encoding='utf-8', xml_declaration=True)

def process_files(file_paths):
    for file_path in file_paths:
        if file_path.lower().endswith('.srt'):
            itt_file = os.path.splitext(file_path)[0] + '.itt'
            try:
                convert_srt_to_itt(file_path, itt_file)
                print(f"Converted: {file_path}\nSaved to: {itt_file}")
            except Exception as e:
                print(f"Failed to convert {file_path}: {e}")

class SRTToITTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SRT to ITT Converter")
        self.setGeometry(100, 100, 400, 200)

        self.label = QLabel("Drag and drop SRT files here or click 'Select Files'", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)

        self.select_button = QPushButton("Select Files", self)
        self.select_button.clicked.connect(self.open_file_dialog)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.select_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
        threading.Thread(target=process_files, args=(file_paths,)).start()

    def open_file_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select SRT Files", "", "SRT Files (*.srt)")
        if file_paths:
            threading.Thread(target=process_files, args=(file_paths,)).start()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = SRTToITTApp()
    main_window.show()
    sys.exit(app.exec_())
