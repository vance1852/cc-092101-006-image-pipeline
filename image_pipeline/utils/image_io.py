from typing import Tuple
from PIL import Image as PILImage
import hashlib
import json
import os
import tempfile
from ..algorithms.core import Image as AlgoImage, Pixel

#: Prefix for temporary files created by interrupted atomic writes.
TEMP_FILE_PREFIX = '.imgwrite-'

def pil_to_algo(pil_img: PILImage.Image) -> AlgoImage:
    if pil_img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
        if 'A' in pil_img.mode:
            pil_img = pil_img.convert('RGBA')
        else:
            pil_img = pil_img.convert('RGB')
    mode = pil_img.mode
    width, height = pil_img.size
    raw = list(pil_img.getdata())
    out: AlgoImage = []
    for y in range(height):
        row = []
        for x in range(width):
            p = raw[y * width + x]
            if mode == 'L':
                row.append(int(p))
            elif mode == 'LA':
                row.append(int(p[0]))
            elif mode == 'RGB':
                row.append([int(p[0]), int(p[1]), int(p[2])])
            elif mode == 'RGBA':
                row.append([int(p[0]), int(p[1]), int(p[2]), int(p[3])])
        out.append(row)
    return out

def algo_to_pil(img: AlgoImage) -> PILImage.Image:
    if not img or not img[0]:
        return PILImage.new('L', (0, 0))
    height = len(img)
    width = len(img[0])
    sample = img[0][0]
    if isinstance(sample, int):
        mode = 'L'
        flat = []
        for row in img:
            flat.extend(row)
        pil_img = PILImage.new(mode, (width, height))
        pil_img.putdata(flat)
    else:
        channels = len(sample)
        if channels == 3:
            mode = 'RGB'
        elif channels == 4:
            mode = 'RGBA'
        else:
            mode = 'RGB'
            new_img = []
            for row in img:
                new_row = []
                for p in row:
                    if len(p) >= 3:
                        new_row.append([p[0], p[1], p[2]])
                    else:
                        g = p[0] if len(p) >= 1 else 0
                        new_row.append([g, g, g])
                new_img.append(new_row)
            img = new_img
        flat = []
        for row in img:
            for p in row:
                flat.append(tuple(p))
        pil_img = PILImage.new(mode, (width, height))
        pil_img.putdata(flat)
    return pil_img

def read_image(path: str) -> AlgoImage:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Image file not found: {path}')
    pil_img = PILImage.open(path)
    return pil_to_algo(pil_img)

def write_image(img: AlgoImage, path: str, fmt: str=None, quality: int=90) -> None:
    pil_img = algo_to_pil(img)
    if fmt is None:
        ext = os.path.splitext(path)[1].lower()
        format_map = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.bmp': 'BMP', '.tif': 'TIFF', '.tiff': 'TIFF', '.webp': 'WEBP'}
        fmt = format_map.get(ext, 'PNG')
    out_dir = os.path.dirname(path)
    if out_dir and (not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)
    if fmt == 'JPEG' and pil_img.mode in ('RGBA', 'LA', 'P'):
        bg = PILImage.new('RGB', pil_img.size, (255, 255, 255))
        if pil_img.mode == 'P':
            pil_img = pil_img.convert('RGBA')
        bg.paste(pil_img, mask=pil_img.split()[-1] if 'A' in pil_img.mode else None)
        pil_img = bg
    elif fmt == 'JPEG' and pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    save_kwargs = {}
    if fmt == 'JPEG':
        save_kwargs['quality'] = max(1, min(95, quality))
        save_kwargs['optimize'] = True
    elif fmt == 'PNG':
        save_kwargs['optimize'] = True
    write_file_atomic(path, lambda tmp_path: pil_img.save(tmp_path, format=fmt, **save_kwargs))


def write_file_atomic(dest_path: str, writer) -> str:
    """Write a file atomically.

    ``writer(tmp_path)`` must fully write ``tmp_path``.  The temp file is
    fsynced, then ``os.replace`` swaps it into place and the directory is
    fsynced.  A crash therefore leaves either the previous contents of
    ``dest_path`` intact or a fully written new file -- never a partially
    written destination.  Temp files left by an interrupted write are
    cleaned up by :func:`cleanup_temp_files`.
    """
    out_dir = os.path.dirname(dest_path) or '.'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=TEMP_FILE_PREFIX, dir=out_dir)
    os.close(fd)
    try:
        writer(tmp_path)
        _fsync_file(tmp_path)
        os.replace(tmp_path, dest_path)
        _fsync_dir(out_dir)
    except BaseException:
        _quiet_remove(tmp_path)
        raise
    return dest_path


def _fsync_file(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _quiet_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def cleanup_temp_files(directory: str) -> int:
    """Remove leftover atomic-write temp files from ``directory`` (top level).

    Returns the number of files removed.
    """
    removed = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0
    for name in names:
        if name.startswith(TEMP_FILE_PREFIX):
            fpath = os.path.join(directory, name)
            try:
                os.remove(fpath)
                removed += 1
            except OSError:
                pass
    return removed


def file_sha256(path: str, chunk_size: int=1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(obj) -> str:
    """Stable hash of a JSON-serialisable object (sorted keys, no whitespace)."""
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def image_size(path: str) -> Tuple[int, int]:
    with PILImage.open(path) as img:
        return img.size

def is_valid_image(path: str) -> bool:
    try:
        with PILImage.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif'}

def find_images(directory: str) -> list:
    if not os.path.isdir(directory):
        return []
    result = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                result.append(path)
    return result
