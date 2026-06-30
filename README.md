# SRT_GEN

基于 Whisper 和 PyQt5 的音视频字幕生成工具。Apple Silicon 上使用 [MLX](https://github.com/ml-explore/mlx) 优化的 `mlx_whisper`，其余平台使用 `openai-whisper`。

---

## 主要功能

- 拖拽或多选音视频文件（支持批量）：`mp3 / wav / m4a / flac / ogg / aac / opus / wma / aiff / amr / alac` 与 `mp4 / mkv / mov / m4v / avi / ts / webm / wmv / flv / mpeg / mpg`
- 多种模型可选，默认 **Large V3 Turbo**（速度约为 large-v3 的数倍，质量接近，推荐）
- 内置**模型管理**：预下载、显示缓存大小、一键删除缓存
- **语言选择**（自动检测 / 中文 / 英语 / 日语…）与**任务选择**（转录 / 翻译成英文）
- 运行设备选择（Apple Silicon 自动用 MLX；其余平台 CPU / CUDA）
- 生成标准 `.srt`，并可选同时导出 Apple `.itt`
- **ffmpeg 已随包内置**，下载安装即用，无需另行安装
- **多线程加速下载模型**，实时显示进度百分比与速度（MB/s）
- 模型缓存：批量处理时只加载一次模型
- 支持拖拽音视频文件到窗口（多文件）

> 命令行批量转 ITT：`python srt2itt.py a.srt b.srt`

---

## 下载安装（发行版）

到 [Releases](https://github.com/wangyaominde/SRT_GEN/releases) 下载对应平台的压缩包：

- **Windows**：`SRT_gen-windows-x64.zip` → 解压后运行文件夹内的 `SRT_gen.exe`
- **macOS（Apple Silicon）**：`SRT_gen-macos-arm64.zip` → 解压得到 `SRT_gen.app`

### macOS 首次打开提示“已损坏 / 无法验证开发者”

发行版未做苹果公证（notarization），从网络下载的应用会被 Gatekeeper 拦截。在终端执行一次即可：

```bash
xattr -cr /路径/SRT_gen.app
```

然后正常双击打开（或在「系统设置 → 隐私与安全性」点“仍要打开”）。

> 当前 macOS 版本为 **Apple Silicon (arm64) 专用**，不支持 Intel Mac。

### 关于模型下载

模型**不随包封装**（`large` 系列单个就有约 3GB，超过 GitHub 单文件上限），首次选用某个模型时会**多线程并行下载并显示进度/速度**，之后缓存复用：

- Apple Silicon：缓存于 `~/.cache/srtgen_models`（下载失败时回退到后端默认的 `~/.cache/huggingface`）
- 其余平台：缓存于 `~/.cache/whisper`

首次使用大模型耗时取决于网络；多线程分块下载会尽量跑满带宽。

---

## 从源码运行

```bash
git clone https://github.com/wangyaominde/SRT_GEN.git
cd SRT_GEN
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 main.py
```

`requirements.txt` 使用平台标记自动选择后端：Apple Silicon 装 `mlx_whisper`，其余平台装 `openai-whisper + torch`。

---

## 本地打包

```bash
pip install pyinstaller
pyinstaller SRT_gen.spec --noconfirm --clean
```

- Windows 产物：`dist/SRT_gen/SRT_gen.exe`
- macOS 产物：`dist/SRT_gen.app`

打包后可冒烟测试（验证 ffmpeg、模型资源、Qt 插件是否齐全，无需下载模型）：

```bash
# macOS
./dist/SRT_gen.app/Contents/MacOS/SRT_gen --selftest
# Windows
dist\SRT_gen\SRT_gen.exe --selftest
```

图标可重新生成：`python scripts/make_icon.py && python scripts/build_icons.py`

---

## 自动构建发布（CI）

`.github/workflows/build.yml` 在以下情况触发：

- 推送 `v*` 标签：构建 Windows / macOS 产物并自动创建 Release 上传
- 手动 `workflow_dispatch`：构建并上传为 Artifact（不创建 Release），便于测试

发布新版本：

```bash
git tag v2.0.0
git push origin v2.0.0
```

---

## 许可证与贡献

欢迎提交 PR 或 issue。
