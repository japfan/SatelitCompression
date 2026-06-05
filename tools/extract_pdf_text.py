import re
import sys
import zlib
from pathlib import Path


def decode_pdf_string(value: str) -> str:
    value = value.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
    value = re.sub(r"\\([nrtbf])", lambda m: {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}[m.group(1)], value)
    value = re.sub(r"\\([0-7]{1,3})", lambda m: chr(int(m.group(1), 8)), value)
    return value


def decode_hex_run(value: str) -> str:
    chars = []
    for token in re.findall(r"[0-9A-Fa-f]{4}", value):
        code = int(token, 16)
        if code == 0x020E:
            chars.append("-")
        elif 0x0000 <= code <= 0x00FF:
            chars.append(chr(code + 29))
        else:
            chars.append(chr(code))
    return "".join(chars)


def extract_text(data: bytes) -> str:
    streams: list[bytes] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        raw = match.group(1)
        try:
            streams.append(zlib.decompress(raw))
        except zlib.error:
            streams.append(raw)

    chunks: list[str] = []
    for stream in streams:
        text = stream.decode("latin-1", errors="ignore")
        for array in re.findall(r"\[(.*?)\]\s*TJ", text, re.S):
            parts = re.findall(r"\((?:\\.|[^\\)])*\)", array)
            if parts:
                chunks.append("".join(decode_pdf_string(part[1:-1]) for part in parts))
            hex_parts = re.findall(r"<([0-9A-Fa-f]+)>", array)
            if hex_parts:
                chunks.append("".join(decode_hex_run(part) for part in hex_parts))
        for item in re.findall(r"\((?:\\.|[^\\)])*\)\s*Tj", text, re.S):
            chunks.append(decode_pdf_string(item[1:-4]))
        for item in re.findall(r"<([0-9A-Fa-f]+)>\s*Tj", text, re.S):
            chunks.append(decode_hex_run(item))

    cleaned = []
    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            cleaned.append(chunk)
    return "\n".join(cleaned)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: extract_pdf_text.py <pdf>")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(extract_text(Path(sys.argv[1]).read_bytes()))
