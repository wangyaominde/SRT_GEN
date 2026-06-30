# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（跨平台）。

构建：
    pyinstaller SRT_gen.spec --noconfirm

产物（onedir，便于可靠加载 torch / mlx 等大型依赖）：
    - Windows : dist/SRT_gen/SRT_gen.exe（+ 同目录依赖）
    - macOS   : dist/SRT_gen.app（Apple Silicon）

关键点：whisper / mlx_whisper 的数据资源（mel_filters.npz、*.tiktoken）、mlx 的
Metal 内核（mlx.metallib）、imageio-ffmpeg 自带的 ffmpeg 二进制、tiktoken 的动态
插件均需显式收集，否则编译通过但运行时崩溃。
"""
import os
import sys
import glob
import platform

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

IS_MAC = sys.platform == 'darwin'
IS_WIN = sys.platform == 'win32'
IS_APPLE_SILICON = IS_MAC and platform.machine() == 'arm64'

datas = [('assets/icon.png', 'assets')]
binaries = []
hiddenimports = []
excludes = []


def _collect_all(pkg):
    d, b, h = collect_all(pkg)
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)


# imageio-ffmpeg 自带的 ffmpeg 二进制（数据文件形式）
datas += collect_data_files('imageio_ffmpeg')

# numpy 2.x 把内部拆到 numpy._core.*，PyInstaller 默认可能漏收 _core 子模块
# （运行时报 No module named 'numpy._core._exceptions'），显式全量收集。
_collect_all('numpy')

# tiktoken 通过命名空间包动态发现编码，需显式声明隐藏导入
hiddenimports += ['tiktoken_ext', 'tiktoken_ext.openai_public']
try:
    hiddenimports += collect_submodules('tiktoken_ext')
except Exception:
    pass

if IS_APPLE_SILICON:
    # MLX 路径：收集 mlx_whisper 资源 + mlx 的 metallib / 动态库 / 扩展
    _collect_all('mlx')
    _collect_all('mlx_whisper')
    # 该平台不使用 openai-whisper；torch 仅被 mlx_whisper.torch_whisper（权重转换，
    # 非转录路径）引用，排除可省去约 400MB。
    excludes += ['whisper', 'torch', 'torchvision', 'torchaudio', 'torchgen']

    # mlx_whisper 依赖 scipy；其 1.18 的 C 扩展默认收不全（运行时报 scipy 损坏）
    _collect_all('scipy')

    # 修复嵌套 rpath：libmlx.dylib 的 rpath 为 @loader_path/../..（= 顶层
    # Frameworks/），它依赖 @rpath/libjaccl.dylib。PyInstaller 只为 libmlx 建了
    # 顶层符号链接，未为 libjaccl 建立，导致运行时 dlopen 失败。这里把 mlx/lib
    # 下除 libmlx 外的 dylib 显式放到包顶层，使其 rpath 能解析到。
    # mlx 0.31+ 是命名空间包（mlx + mlx-metal 共享 mlx/ 目录），__file__ 为 None，
    # 用 __path__ 定位实际目录。
    import mlx as _mlx
    for _root in list(getattr(_mlx, '__path__', [])):
        for _dylib in glob.glob(os.path.join(_root, 'lib', '*.dylib')):
            if os.path.basename(_dylib) != 'libmlx.dylib':
                binaries.append((_dylib, '.'))
else:
    # whisper + torch 路径
    datas += collect_data_files('whisper')   # whisper/assets/*
    _collect_all('torch')                    # 含 torch.jit 所需的 .py 源与动态库
    hiddenimports += ['whisper']
    excludes += ['mlx', 'mlx_whisper']        # 该平台不使用 MLX

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SRT_gen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI 程序，不弹控制台窗口
    disable_windowed_traceback=False,
    icon='assets/icon.ico' if IS_WIN else ('assets/icon.icns' if IS_MAC else None),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SRT_gen',
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name='SRT_gen.app',
        icon='assets/icon.icns',
        bundle_identifier='com.wangyaominde.srtgen',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleName': 'SRT_gen',
            'CFBundleDisplayName': 'SRT 字幕生成器',
            'CFBundleShortVersionString': '2.1.0',
            'CFBundleVersion': '2.1.0',
            'LSMinimumSystemVersion': '11.0',
        },
    )
