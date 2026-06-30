"""SRT → ITT（Apple iTunes Timed Text）转换。

提供可被 main.py 复用的纯函数（parse_srt / convert_srt_to_itt），并保留一个
带有结果反馈的独立 GUI 以及命令行入口：

    python srt2itt.py a.srt b.srt        # 命令行批量转换
    python srt2itt.py                    # 打开 GUI

相比旧版本修复了：
- 致命的贪婪正则（旧版把整个 SRT 塌缩成一条字幕）；改为按空行切块解析。
- 仅 UTF-8 读取导致 GBK/Big5 文件崩溃；改为多编码回退。
- 生成的 ITT 缺少 head/styling/layout/region/timeBase；改为输出合规骨架。
- 后台线程仅 print、GUI 无反馈；改为信号回传并更新界面。
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

# 时间行：00:00:01,000 --> 00:00:03,000（毫秒分隔符容忍 , 或 .）
_TIME_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)

# 读取 SRT 时的编码回退链：utf-8-sig 处理 BOM，gb18030 覆盖 GBK/GB2312，
# big5 覆盖繁体，latin-1 兜底（可解码任意字节）。
_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "big5", "cp1252", "latin-1"]


def read_text_with_fallback(path):
    """读取文本文件，按编码回退链尝试解码。"""
    with open(path, "rb") as fh:
        raw = fh.read()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_srt(content):
    """把 SRT 文本解析为 [(start, end, text), ...]。

    按空行切分字幕块，每块内定位含 `-->` 的时间行，其后的所有行为文本。
    序号行可有可无；对缺序号、多行文本、CRLF、BOM 均健壮。
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    entries = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.split("\n")
        timing_idx = None
        start = end = None
        for i, line in enumerate(lines):
            m = _TIME_RE.search(line)
            if m:
                timing_idx = i
                start, end = m.group(1), m.group(2)
                break
        if timing_idx is None:
            continue
        text = "\n".join(lines[timing_idx + 1:]).strip()
        entries.append((start, end, text))
    return entries


def _to_itt_time(srt_time):
    """SRT 时间（hh:mm:ss,ms）转 ITT 时间（hh:mm:ss.ms）。"""
    return srt_time.replace(",", ".")


def _set_multiline_text(p, text):
    """把可能含换行的文本写入 <p>，行间用 <br/> 分隔。"""
    parts = text.split("\n")
    p.text = parts[0] if parts else ""
    for extra in parts[1:]:
        br = ET.SubElement(p, "br")
        br.tail = extra


def build_itt_tree(entries, lang="zh"):
    """根据解析结果构建合规的 ITT ElementTree。"""
    tt = ET.Element("tt", {
        "xmlns": "http://www.w3.org/ns/ttml",
        "xmlns:tts": "http://www.w3.org/ns/ttml#styling",
        "xmlns:ttm": "http://www.w3.org/ns/ttml#metadata",
        "xmlns:ttp": "http://www.w3.org/ns/ttml#parameter",
        "ttp:timeBase": "media",
        "xml:lang": lang or "",
    })

    head = ET.SubElement(tt, "head")
    styling = ET.SubElement(head, "styling")
    ET.SubElement(styling, "style", {
        "xml:id": "basic",
        "tts:fontFamily": "sansSerif",
        "tts:fontSize": "100%",
        "tts:color": "white",
        "tts:textAlign": "center",
    })
    layout = ET.SubElement(head, "layout")
    ET.SubElement(layout, "region", {
        "xml:id": "bottom",
        "tts:origin": "10% 80%",
        "tts:extent": "80% 20%",
        "tts:displayAlign": "after",
        "tts:textAlign": "center",
    })

    body = ET.SubElement(tt, "body")
    div = ET.SubElement(body, "div")
    for start, end, text in entries:
        p = ET.SubElement(div, "p", {
            "begin": _to_itt_time(start),
            "end": _to_itt_time(end),
            "region": "bottom",
            "style": "basic",
        })
        _set_multiline_text(p, text)

    return ET.ElementTree(tt)


def convert_srt_to_itt(srt_file, itt_file, lang="zh"):
    """把单个 SRT 文件转换为 ITT 文件。返回写入的字幕条数。"""
    content = read_text_with_fallback(srt_file)
    entries = parse_srt(content)
    if not entries:
        raise ValueError("未解析到任何字幕条目，请确认这是有效的 SRT 文件")
    tree = build_itt_tree(entries, lang=lang)
    ET.indent(tree, space="  ", level=0)
    tree.write(itt_file, encoding="utf-8", xml_declaration=True)
    return len(entries)


def process_files(file_paths, lang="zh"):
    """批量转换，返回 [(srt_path, itt_path_or_None, error_or_None), ...]。"""
    results = []
    for file_path in file_paths:
        if not file_path.lower().endswith(".srt"):
            continue
        itt_file = os.path.splitext(file_path)[0] + ".itt"
        try:
            convert_srt_to_itt(file_path, itt_file, lang=lang)
            results.append((file_path, itt_file, None))
        except Exception as e:  # noqa: BLE001 - 汇总错误供调用方展示
            results.append((file_path, None, str(e)))
    return results


# ----------------------------- GUI（可选，独立运行） -----------------------------

def _run_gui():
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
        QVBoxLayout, QWidget,
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QDragEnterEvent, QDropEvent

    class ConvertWorker(QThread):
        done = pyqtSignal(list)

        def __init__(self, file_paths):
            super().__init__()
            self.file_paths = file_paths

        def run(self):
            self.done.emit(process_files(self.file_paths))

    class SRTToITTApp(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("SRT → ITT 转换器")
            self.setGeometry(100, 100, 440, 200)
            self.worker = None

            self.label = QLabel("拖拽 SRT 文件到此处，或点击下方按钮选择", self)
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setWordWrap(True)

            self.select_button = QPushButton("选择 SRT 文件", self)
            self.select_button.clicked.connect(self.open_file_dialog)

            layout = QVBoxLayout()
            layout.addWidget(self.label)
            layout.addWidget(self.select_button)
            central = QWidget()
            central.setLayout(layout)
            self.setCentralWidget(central)
            self.setAcceptDrops(True)

        def dragEnterEvent(self, event: QDragEnterEvent):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dropEvent(self, event: QDropEvent):
            paths = [u.toLocalFile() for u in event.mimeData().urls()]
            self.start_conversion(paths)

        def open_file_dialog(self):
            paths, _ = QFileDialog.getOpenFileNames(
                self, "选择 SRT 文件", "", "SRT 文件 (*.srt)")
            if paths:
                self.start_conversion(paths)

        def start_conversion(self, paths):
            srt_paths = [p for p in paths if p.lower().endswith(".srt")]
            if not srt_paths:
                self.label.setText("请选择 .srt 文件")
                return
            self.select_button.setEnabled(False)
            self.label.setText(f"正在转换 {len(srt_paths)} 个文件...")
            self.worker = ConvertWorker(srt_paths)
            self.worker.done.connect(self.on_done)
            self.worker.start()

        def on_done(self, results):
            self.select_button.setEnabled(True)
            ok = [r for r in results if r[2] is None]
            failed = [r for r in results if r[2] is not None]
            msg = f"完成：成功 {len(ok)} 个，失败 {len(failed)} 个"
            if failed:
                first = os.path.basename(failed[0][0])
                msg += f"\n（{first}：{failed[0][2]}）"
            self.label.setText(msg)

    app = QApplication(sys.argv)
    win = SRTToITTApp()
    win.show()
    sys.exit(app.exec_())


def _run_cli(paths):
    results = process_files(paths)
    for srt_path, itt_path, err in results:
        if err is None:
            print(f"已转换: {srt_path} -> {itt_path}")
        else:
            print(f"失败: {srt_path}: {err}", file=sys.stderr)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.lower().endswith(".srt")]
    if args:
        _run_cli(args)
    else:
        _run_gui()
