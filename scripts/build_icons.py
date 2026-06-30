#!/usr/bin/env python3
"""把 assets/icon.png 转换为 Windows .ico 和 macOS .icns。

用法：
    python scripts/build_icons.py

- icon.ico  : 由 Pillow 生成（任意平台可用）
- icon.icns : 优先用 macOS 的 iconutil；不可用时回退到 Pillow（若其支持 icns）

CI 直接使用仓库中已生成的图标文件，无需在构建时重跑本脚本。
"""
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNG = os.path.join(ROOT, "assets", "icon.png")
ICO = os.path.join(ROOT, "assets", "icon.ico")
ICNS = os.path.join(ROOT, "assets", "icon.icns")


def make_ico():
    img = Image.open(PNG).convert("RGBA")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ICO, format="ICO", sizes=sizes)
    print(f"wrote {ICO}")


def make_icns():
    if shutil.which("iconutil") and shutil.which("sips"):
        with tempfile.TemporaryDirectory() as tmp:
            iconset = os.path.join(tmp, "icon.iconset")
            os.makedirs(iconset)
            specs = [
                (16, "16x16"), (32, "16x16@2x"),
                (32, "32x32"), (64, "32x32@2x"),
                (128, "128x128"), (256, "128x128@2x"),
                (256, "256x256"), (512, "256x256@2x"),
                (512, "512x512"), (1024, "512x512@2x"),
            ]
            for px, name in specs:
                out = os.path.join(iconset, f"icon_{name}.png")
                subprocess.run(["sips", "-z", str(px), str(px), PNG, "--out", out],
                               check=True, capture_output=True)
            subprocess.run(["iconutil", "-c", "icns", iconset, "-o", ICNS], check=True)
            print(f"wrote {ICNS}")
            return
    # 回退：Pillow（部分版本支持 icns）
    try:
        Image.open(PNG).convert("RGBA").save(ICNS, format="ICNS")
        print(f"wrote {ICNS} (Pillow)")
    except Exception as e:
        print(f"skip icns (no iconutil and Pillow icns failed: {e})", file=sys.stderr)


if __name__ == "__main__":
    if not os.path.exists(PNG):
        sys.exit("assets/icon.png missing — run scripts/make_icon.py first")
    make_ico()
    make_icns()
