"""把一个目录里的文档整理成 Markdown。

支持 .txt、.md、.docx、.doc、.pdf，以及 .zip。
.rar 只在系统能找到 7-Zip 时解开。抽不出正文的文件跳过，并写进输出目录的 skipped.txt。
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path

import olefile
from docx import Document
from pypdf import PdfReader

TEXT_SUFFIXES = {".txt", ".md", ".docx", ".doc", ".pdf"}
ARCHIVE_SUFFIXES = {".zip", ".rar"}
_PAGE_NUMBER = re.compile(
    r"^(第\s*)?\d+\s*页$|^[-—]?\s*\d+\s*[-—]?$|^page\s+\d+$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """去掉控制符、单独的页码和多余空行。表格抽出来的短字段会保留。"""
    kept: list[str] = []
    for char in text:
        code = ord(char)
        if char in "\n\t" or 32 <= code <= 126 or 0x4E00 <= code <= 0x9FFF:
            kept.append(char)
        elif 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF or 0x2010 <= code <= 0x2027:
            kept.append(char)
    text = "".join(kept)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or _PAGE_NUMBER.match(line):
            continue
        if len(line) <= 2 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", line):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _ascii_letters(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def extract_doc(path: Path) -> str:
    """读取老格式 .doc。

    这部分是针对特定 .doc 写的，不是完整的 Word 解析。换一批文件时按实际情况改。
    """
    ole = olefile.OleFileIO(str(path))
    try:
        word = ole.openstream("WordDocument").read()
    finally:
        ole.close()
    fc_min, fc_mac = struct.unpack_from("<II", word, 0x18)
    if not 0 <= fc_min < fc_mac <= len(word):
        raise ValueError("文档结构里没有可用的正文范围")
    span = word[fc_min:fc_mac]
    ccp_text = _word_ccp_text(word)
    if ccp_text <= 0:
        ccp_text = len(span)
    utf_raw = span[: ccp_text * 2]
    # 中文正文的高字节很多会大于 0x80；英文模板几乎没有。
    if sum(byte >= 0x80 for byte in utf_raw) >= 20:
        text = utf_raw.decode("utf-16le", "ignore")
        nulls = text.find("\x00\x00")
        if nulls >= 20:
            text = text[:nulls]
    else:
        byte_raw = span[:ccp_text]
        as_latin = byte_raw.decode("latin1", "ignore")
        as_gbk = byte_raw.decode("gbk", "ignore")
        text = as_gbk if _ascii_letters(as_gbk) > _ascii_letters(as_latin) else as_latin
    return clean_text(text.replace("\x07", "\n").replace("\x00", ""))


def _word_ccp_text(word: bytes) -> int:
    """FIB 里主文档的字符数。算不准时返回 0，调用方改用整个正文范围。"""
    if len(word) < 0x24:
        return 0
    csw = struct.unpack_from("<H", word, 0x20)[0]
    offset = 0x22 + csw * 2
    if offset + 2 > len(word):
        return 0
    cslw = struct.unpack_from("<H", word, offset)[0]
    if cslw < 4 or offset + 2 + 16 > len(word):
        return 0
    return int(struct.unpack_from("<I", word, offset + 2 + 12)[0])


def extract_docx(path: Path) -> str:
    document = Document(str(path))
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return clean_text("\n".join(parts))


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return clean_text(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".docx":
        return extract_docx(path)
    if suffix == ".doc":
        return extract_doc(path)
    if suffix == ".pdf":
        return extract_pdf(path)
    raise ValueError(f"不支持的文件类型：{suffix}")


def _expand_zip(archive: Path, work: Path, source_root: Path) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    label = archive.relative_to(source_root)
    with zipfile.ZipFile(archive) as package:
        for info in package.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if name.suffix.lower() not in TEXT_SUFFIXES:
                continue
            target = work / label.stem / name.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(package.read(info))
            found.append((target, label.parent / label.stem / name.name))
    return found


def _expand_rar(archive: Path, work: Path, source_root: Path) -> list[tuple[Path, Path]]:
    seven = shutil.which("7z") or shutil.which("7za")
    if not seven:
        raise ValueError("未安装 7-Zip，无法解开 rar")
    label = archive.relative_to(source_root)
    destination = work / label.stem
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [seven, "x", "-y", f"-o{destination}", str(archive)],
        check=True,
        capture_output=True,
    )
    found: list[tuple[Path, Path]] = []
    for path in destination.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            relative = label.parent / label.stem / path.relative_to(destination)
            found.append((path, relative))
    return found


def iter_documents(source: Path, work: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    documents: list[tuple[Path, Path]] = []
    skipped: list[str] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        suffix = path.suffix.lower()
        relative = path.relative_to(source)
        try:
            if suffix in TEXT_SUFFIXES:
                documents.append((path, relative))
            elif suffix == ".zip":
                documents.extend(_expand_zip(path, work, source))
            elif suffix == ".rar":
                documents.extend(_expand_rar(path, work, source))
        except (ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
            skipped.append(f"{relative.as_posix()}\t{exc}")
    return documents, skipped


def _markdown(title: str, source: str, text: str) -> str:
    return f"# {title}\n\n来源：{source}\n\n{text}\n"


def prepare_directory(source: Path, output: Path, min_chars: int = 40) -> tuple[list[Path], list[str]]:
    if not source.exists():
        raise FileNotFoundError(f"找不到文档目录：{source}")
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped: list[str] = []
    used: set[Path] = set()
    with tempfile.TemporaryDirectory() as temporary:
        documents, archive_skips = iter_documents(source, Path(temporary))
        skipped.extend(archive_skips)
        for path, relative in documents:
            try:
                text = extract_text(path)
            except Exception as exc:
                skipped.append(f"{relative.as_posix()}\t{exc}")
                continue
            if len(text) < min_chars:
                skipped.append(f"{relative.as_posix()}\t抽不出足够正文")
                continue
            destination = output / relative.with_suffix(".md")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination in used or destination.exists():
                destination = destination.with_name(f"{destination.stem}-{len(used)}{destination.suffix}")
            used.add(destination)
            destination.write_text(
                _markdown(relative.stem, relative.as_posix(), text),
                encoding="utf-8",
            )
            written.append(destination)
    (output / "skipped.txt").write_text("\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8")
    return written, skipped


def main() -> None:
    from hybridrag.config import PREPARED_DIR, ROOT

    parser = argparse.ArgumentParser(description="把目录中的文档整理成 Markdown 语料")
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "data" / "local-dataset")
    parser.add_argument("-o", "--output", type=Path, default=PREPARED_DIR)
    args = parser.parse_args()
    written, skipped = prepare_directory(args.source, args.output)
    print(f"写出 {len(written)} 篇，跳过 {len(skipped)} 个")
    for line in skipped:
        print(f"跳过\t{line}")


if __name__ == "__main__":
    main()
