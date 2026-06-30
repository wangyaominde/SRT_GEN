"""带进度/速度显示的多线程（分块并行）模型下载器。

对外提供：
- parallel_download(url, dest, on_progress, connections): 通用多连接分块下载，
  服务器不支持 Range 时自动回退单流。
- ensure_whisper_model(size, ...): 把 openai-whisper 的 .pt 下到 ~/.cache/whisper，
  下完交由 whisper.load_model 校验（sha 不符会自动重下）。
- ensure_mlx_model(size, ...): 把 mlx-community/whisper-{size}-mlx 仓库文件下到本地目录，
  返回该目录路径，供 mlx_whisper.transcribe(path_or_hf_repo=<dir>) 离线加载。

设计原则：均为「优化层」，任何异常都应由调用方捕获并回退到后端自带的下载逻辑，
绝不因下载加速失败而影响转录本身。
"""
import os
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_CHUNK = 1 << 20          # 1 MiB
_DEFAULT_CONNECTIONS = 8
_TIMEOUT = 45             # 单次读/连接超时（秒）：超过即判定卡住并续传重试
_RETRIES = 6             # 每个分块连续「无进展」重试上限
_UA = {'User-Agent': 'SRT_gen/2.x'}


def _resolve(url, timeout=_TIMEOUT):
    """探测大小与是否支持 Range（用 bytes=0-0 单字节请求）。

    返回 (total_bytes, ranges_supported)；total 为 0 表示未知。
    注意：不复用重定向后的 CDN URL —— HuggingFace 的 resolve 会重定向到带签名、
    短时效的 CDN 链接，复用到多个分块请求上会失败；每个分块需各自请求原始 URL，
    由 urllib 每次重新重定向。
    """
    try:
        req = urllib.request.Request(url, headers={**_UA, 'Range': 'bytes=0-0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = getattr(r, 'status', r.getcode())
            cr = r.headers.get('Content-Range')
            if status == 206 and cr and '/' in cr:
                try:
                    return int(cr.rsplit('/', 1)[1]), True
                except ValueError:
                    pass
            cl = r.headers.get('Content-Length')
            return (int(cl) if cl and cl.isdigit() else 0), False
    except Exception:
        return 0, False


def _split(total, parts):
    step = total // parts
    ranges = []
    start = 0
    for i in range(parts):
        end = total - 1 if i == parts - 1 else (start + step - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges


def _single_stream(url, part_path, total, on_progress, timeout=_TIMEOUT, retries=_RETRIES):
    """单流下载（服务器不支持 Range）。卡住/失败则重试（从头）。"""
    attempt = 0
    while True:
        try:
            t0 = time.time()
            done = 0
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part_path, 'wb') as f:
                while True:
                    buf = r.read(_CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if on_progress:
                        el = time.time() - t0
                        on_progress(done, total or done, done / el if el > 0 else 0)
            return
        except Exception:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(min(2 * attempt, 8))


def parallel_download(url, dest, on_progress=None, connections=_DEFAULT_CONNECTIONS,
                      timeout=_TIMEOUT, retries=_RETRIES):
    """多连接分块下载到 dest（先写 dest.part 再原子替换）。

    抗卡住设计：
    - 先解析一次最终 URL，避免每个分块重复重定向/限流；
    - 每个分块独立「断点续传 + 重试」：连接中断或读卡住（超过 timeout 无数据）后，
      从已写位置继续，只要仍有进展就不消耗重试次数；连续无进展超过 retries 次才放弃；
    - 不会因单个分块失败而把已下载内容全部作废重来。

    on_progress(downloaded_bytes, total_bytes, speed_bytes_per_sec)。
    服务器不支持 Range 或大小未知时回退为单流下载。
    """
    part = dest + '.part'
    total, ranges_ok = _resolve(url, timeout=timeout)

    if not (ranges_ok and total > _CHUNK * 2 and connections > 1):
        _single_stream(url, part, total, on_progress, timeout=timeout, retries=retries)
        os.replace(part, dest)
        return dest

    # 预分配文件，多线程各写一段（区间不重叠，分别打开句柄安全）
    with open(part, 'wb') as f:
        f.truncate(total)

    lock = threading.Lock()
    state = {'done': 0, 't0': time.time()}

    def report(delta):
        if not on_progress:
            return
        with lock:
            state['done'] += delta
            el = time.time() - state['t0']
            on_progress(state['done'], total, state['done'] / el if el > 0 else 0)

    def worker(start, end):
        pos = start
        attempt = 0
        while pos <= end:
            before = pos
            try:
                req = urllib.request.Request(
                    url, headers={**_UA, 'Range': f'bytes={pos}-{end}'})
                with urllib.request.urlopen(req, timeout=timeout) as r, open(part, 'r+b') as f:
                    f.seek(pos)
                    while True:
                        buf = r.read(_CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        pos += len(buf)
                        report(len(buf))
            except Exception:
                pass
            if pos > end:
                return                      # 本分块完成
            if pos > before:
                attempt = 0                 # 有进展 → 重置重试计数，继续续传
            else:
                attempt += 1
                if attempt > retries:
                    raise IOError(f'分块 {start}-{end} 多次重试仍无进展')
                time.sleep(min(2 * attempt, 8))

    try:
        with ThreadPoolExecutor(max_workers=connections) as ex:
            futures = [ex.submit(worker, s, e) for s, e in _split(total, connections)]
            for fu in futures:
                fu.result()  # 任一分块彻底失败则抛出，由调用方回退到后端自带下载
    except Exception:
        try:
            os.remove(part)
        except OSError:
            pass
        raise

    os.replace(part, dest)
    return dest


def _whisper_root():
    return os.path.expanduser('~/.cache/whisper')


def _mlx_root():
    return os.path.expanduser('~/.cache/srtgen_models')


def _hf_hub_dir(repo_id):
    name = 'models--' + repo_id.replace('/', '--')
    return os.path.join(os.path.expanduser('~/.cache/huggingface/hub'), name)


def mlx_cache_dir(repo_id):
    return os.path.join(_mlx_root(), repo_id.split('/')[-1])


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def whisper_cache_path(whisper_name):
    """返回 whisper {name}.pt 的缓存路径；whisper 未安装/未知名返回 None。"""
    try:
        import whisper
    except Exception:
        return None
    url = getattr(whisper, '_MODELS', {}).get(whisper_name)
    if not url:
        return None
    return os.path.join(_whisper_root(), os.path.basename(url))


def model_cache_info(apple, mlx_repo, whisper_name):
    """返回 (是否已缓存, 字节数)。"""
    if apple:
        d = mlx_cache_dir(mlx_repo)
        if os.path.isdir(d) and (os.path.exists(os.path.join(d, 'weights.npz'))
                                 or os.path.exists(os.path.join(d, 'weights.safetensors'))):
            return True, dir_size(d)
        hub = _hf_hub_dir(mlx_repo)
        if os.path.isdir(hub):
            return True, dir_size(hub)
        return False, 0
    p = whisper_cache_path(whisper_name)
    if p and os.path.exists(p):
        return True, os.path.getsize(p)
    return False, 0


def delete_model_cache(apple, mlx_repo, whisper_name):
    """删除模型缓存，返回释放的字节数。"""
    import shutil
    freed = 0
    if apple:
        for d in (mlx_cache_dir(mlx_repo), _hf_hub_dir(mlx_repo)):
            if os.path.isdir(d):
                freed += dir_size(d)
                shutil.rmtree(d, ignore_errors=True)
    else:
        p = whisper_cache_path(whisper_name)
        if p and os.path.exists(p):
            freed += os.path.getsize(p)
            try:
                os.remove(p)
            except OSError:
                pass
    return freed


def ensure_whisper_model(whisper_name, on_progress=None, on_start=None,
                         connections=_DEFAULT_CONNECTIONS):
    """确保 openai-whisper 的 {name}.pt 已在 ~/.cache/whisper。

    已存在则直接返回（由 whisper.load_model 负责 sha 校验/必要时重下）。
    返回 .pt 路径；无法处理（未知模型名）时返回 None 让后端自行下载。
    """
    import whisper
    models = getattr(whisper, '_MODELS', {})
    if whisper_name not in models:
        return None
    url = models[whisper_name]
    root = _whisper_root()
    dest = os.path.join(root, os.path.basename(url))
    if os.path.exists(dest):
        return dest
    os.makedirs(root, exist_ok=True)
    if on_start:
        on_start()
    parallel_download(url, dest, on_progress=on_progress, connections=connections)
    return dest


def ensure_mlx_model(repo_id, on_progress=None, on_start=None,
                     connections=_DEFAULT_CONNECTIONS):
    """确保指定 HF 仓库的文件已下到本地目录，返回该目录供 mlx_whisper 离线加载。

    任何失败应由调用方捕获并回退到传 repo id 让后端自行下载。
    """
    from huggingface_hub import HfApi, hf_hub_url

    target_dir = mlx_cache_dir(repo_id)
    info = HfApi().model_info(repo_id, files_metadata=True)
    sibs = [(s.rfilename, int(getattr(s, 'size', 0) or 0)) for s in info.siblings]

    def ok(fn, sz):
        p = os.path.join(target_dir, fn)
        return os.path.exists(p) and (sz == 0 or os.path.getsize(p) == sz)

    if sibs and all(ok(fn, sz) for fn, sz in sibs):
        return target_dir  # 已缓存

    total = sum(sz for _, sz in sibs) or 0
    if on_start:
        on_start()

    base = 0
    for fn, sz in sibs:
        dest = os.path.join(target_dir, fn)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if ok(fn, sz):
            base += sz
            continue
        url = hf_hub_url(repo_id, fn)
        b0 = base

        def cb(done, _t, speed, _b0=b0):
            if on_progress:
                on_progress(_b0 + done, total or (_b0 + done), speed)

        parallel_download(url, dest, on_progress=cb if on_progress else None,
                          connections=connections)
        base += sz

    return target_dir
