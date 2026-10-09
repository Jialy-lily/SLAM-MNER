#!/usr/bin/env python3
import ast
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".sh"}

# Optional: set SLAM_PRIVATE_WORDS to a comma-separated list of identifiers
# that must not appear in the release.
local_private_words = [
    w.strip() for w in os.environ.get("SLAM_PRIVATE_WORDS", "").split(",") if w.strip()
]
contact_words = ["we" + "chat", "wei" + "xin", "q" + "q"]
PUBLIC_EMAILS = {"zhaoyaru@uir.edu.cn"}
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

FORBIDDEN_PATTERNS = [
    re.compile(r"/Users/[^\s'\"]+"),
    re.compile(r"/(?:home|workspace)/[^\s'\"]+"),
    re.compile(rf"\b({'|'.join(contact_words + local_private_words)})\b", re.IGNORECASE),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
GENERATED_NAMES = {"__pycache__", ".DS_Store", "__MACOSX"}
GENERATED_PREFIXES = ("._",)


def iter_release_files():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts:
            continue
        if any(part in GENERATED_NAMES for part in path.parts) or path.name.startswith(GENERATED_PREFIXES):
            raise AssertionError(f"generated file should not be released: {rel}")
        if path.is_file():
            yield path


def check_python_syntax(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def check_text_for_private_info(path):
    if path == Path(__file__).resolve():
        return
    text = path.read_text(encoding="utf-8")
    for email in EMAIL_PATTERN.findall(text):
        if email.lower() not in PUBLIC_EMAILS:
            raise AssertionError(f"unlisted contact email in {path.relative_to(ROOT)}: {email}")
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(text)
        if match:
            raise AssertionError(f"possible private information in {path.relative_to(ROOT)}: {match.group(0)}")


def check_sample_record():
    sample_path = ROOT / "examples" / "sample_record.json"
    records = json.loads(sample_path.read_text(encoding="utf-8"))
    assert isinstance(records, list) and records, "sample_record.json must contain a non-empty list"
    required = {"id", "text", "image_paths", "gold_entities"}
    for record in records:
        assert required.issubset(record), f"missing keys in sample record: {record}"
        assert isinstance(record["image_paths"], list), "image_paths must be a list"
        assert isinstance(record["gold_entities"], list), "gold_entities must be a list"


def main():
    files = list(iter_release_files())
    for path in files:
        if path.suffix == ".py":
            check_python_syntax(path)
        if path.suffix in TEXT_SUFFIXES:
            check_text_for_private_info(path)
    check_sample_record()
    print(f"Release check passed: {len(files)} files scanned.")


if __name__ == "__main__":
    main()
