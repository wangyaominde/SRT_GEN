#!/usr/bin/env python3
"""生成应用图标 assets/icon.png（1024x1024）。

用法：
    python scripts/make_icon.py

仅依赖 Pillow。生成的 PNG 再由 scripts/build_icons.py 转换为 .ico / .icns，
或在 CI 中按平台转换。重新生成时直接运行本脚本即可。
"""
import os
import math

from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icon.png")

# 候选字体（按系统常见路径回退）
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def rounded_mask(size, radius):
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def main():
    # 对角渐变背景：靛蓝 -> 紫罗兰
    top = (79, 70, 229)     # #4F46E5
    bottom = (124, 58, 237)  # #7C3AED
    bg = Image.new("RGB", (SIZE, SIZE))
    px = bg.load()
    for y in range(SIZE):
        for x in range(SIZE):
            t = (x + y) / (2 * (SIZE - 1))
            px[x, y] = lerp(top, bottom, t)

    draw = ImageDraw.Draw(bg)

    # 顶部柔光
    glow = Image.new("L", (SIZE, SIZE), 0)
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-SIZE * 0.3, -SIZE * 0.55, SIZE * 1.3, SIZE * 0.5], fill=70)
    bg = Image.composite(Image.new("RGB", (SIZE, SIZE), (255, 255, 255)), bg, glow)
    draw = ImageDraw.Draw(bg)

    # 中部播放三角（字幕来源 = 音视频）
    cx, cy = SIZE // 2, int(SIZE * 0.40)
    r = int(SIZE * 0.155)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=int(SIZE * 0.018))
    tri = int(r * 0.52)
    draw.polygon(
        [(cx - tri * 0.55, cy - tri), (cx - tri * 0.55, cy + tri), (cx + tri * 0.95, cy)],
        fill=(255, 255, 255),
    )

    # 底部三条字幕条（白/半透明），模拟字幕行
    bar_h = int(SIZE * 0.058)
    gap = int(SIZE * 0.038)
    base_y = int(SIZE * 0.605)
    widths = [0.62, 0.50, 0.40]
    for i, wf in enumerate(widths):
        w = int(SIZE * wf)
        x0 = (SIZE - w) // 2
        y0 = base_y + i * (bar_h + gap)
        alpha = 255 if i == 0 else (200 if i == 1 else 150)
        bar = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bar)
        bd.rounded_rectangle([x0, y0, x0 + w, y0 + bar_h], radius=bar_h // 2,
                             fill=(255, 255, 255, alpha))
        bg = Image.alpha_composite(bg.convert("RGBA"), bar).convert("RGB")
    draw = ImageDraw.Draw(bg)

    # 左上角小标 "SRT"
    font = load_font(int(SIZE * 0.085))
    draw.text((int(SIZE * 0.085), int(SIZE * 0.075)), "SRT", font=font,
              fill=(255, 255, 255))

    # 圆角裁切
    radius = int(SIZE * 0.225)
    mask = rounded_mask(SIZE, radius)
    out = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    out.paste(bg, (0, 0), mask)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.save(OUT)
    print(f"wrote {OUT} ({SIZE}x{SIZE})")


if __name__ == "__main__":
    main()
