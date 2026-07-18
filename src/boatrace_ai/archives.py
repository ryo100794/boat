from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def extract_lzh(path: str | Path) -> list[tuple[str, bytes]]:
    archive = Path(path)
    extracted = _extract_with_lhafile(archive)
    if extracted:
        return extracted
    for command in ("lha", "7z", "bsdtar"):
        if shutil.which(command):
            extracted = _extract_with_command(archive, command)
            if extracted:
                return extracted
    return []


def _extract_with_lhafile(path: Path) -> list[tuple[str, bytes]]:
    try:
        import lhafile
    except ImportError:
        return []
    try:
        result = []
        with lhafile.Lhafile(str(path)) as archive:
            for info in archive.infolist():
                if info.filename.endswith("/"):
                    continue
                result.append((info.filename, archive.read(info.filename)))
        return result
    except Exception:
        return []


def _extract_with_command(path: Path, command: str) -> list[tuple[str, bytes]]:
    with tempfile.TemporaryDirectory(prefix="boatrace-lzh-") as tmp:
        target = Path(tmp)
        if command == "lha":
            cmd = [command, "xq", str(path)]
            cwd = target
        elif command == "7z":
            cmd = [command, "x", "-y", f"-o{target}", str(path)]
            cwd = None
        elif command == "bsdtar":
            cmd = [command, "-xf", str(path), "-C", str(target)]
            cwd = None
        else:
            return []
        try:
            subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return []

        result = []
        for file_path in sorted(p for p in target.rglob("*") if p.is_file()):
            result.append((file_path.relative_to(target).as_posix(), file_path.read_bytes()))
        return result


def decode_japanese_text(payload: bytes) -> str:
    for encoding in ("cp932", "shift_jis", "utf-8"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("cp932", errors="replace")
