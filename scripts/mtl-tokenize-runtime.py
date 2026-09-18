#!/usr/bin/env python3
import argparse
import ast
import importlib.util
import json
import struct
from pathlib import Path

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("chatterbox_official_tokenizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    if node is None:
        raise RuntimeError(f"{name} not found in {path}")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("punc", "tokenize"), required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--tts-source", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--cangjie", required=True)
    p.add_argument("--dicta-model", required=True)
    p.add_argument("--language", default="")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    text = Path(a.input).read_text(encoding="utf-8")
    if a.mode == "punc":
        normalized = load_function(Path(a.tts_source).resolve(), "punc_norm")(text)
        Path(a.output).write_text(normalized, encoding="utf-8", newline="")
        return 0
    source = Path(a.source).resolve()
    tokenizer_path = Path(a.tokenizer).resolve()
    cangjie_path = Path(a.cangjie).resolve()
    dicta_model = Path(a.dicta_model).resolve()
    module = load_module(source)
    cangjie_data = json.loads(cangjie_path.read_text(encoding="utf-8"))
    def load_cangjie(self, model_dir=None):
        self.word2cj = {}
        self.cj2word = {}
        for entry in cangjie_data:
            word, code = entry.split("\t")[:2]
            self.word2cj[word] = code
            self.cj2word.setdefault(code, []).append(word)
    module.ChineseCangjieConverter._load_cangjie_mapping = load_cangjie
    if a.language == "he":
        from dicta_onnx import Dicta
        module._dicta = Dicta(str(dicta_model))
    elif a.language == "ja":
        import pykakasi
        module._kakasi = pykakasi.kakasi()
    elif a.language == "ru":
        from russian_text_stresser.text_stresser import RussianTextStresser
        module._russian_stresser = RussianTextStresser()
    elif a.language == "zh":
        import spacy_pkuseg
    tokenizer = module.MTLTokenizer(str(tokenizer_path))
    class CaptureTokenizer:
        def __init__(self, inner):
            self.inner = inner
            self.input = None
        def encode(self, text, *args, **kwargs):
            self.input = text
            return self.inner.encode(text, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.inner, name)
    capture = CaptureTokenizer(tokenizer.tokenizer)
    tokenizer.tokenizer = capture
    ids = tokenizer.encode(text, language_id=a.language)
    if capture.input is None:
        raise RuntimeError("official tokenizer encode did not execute")
    tokenizer_input = capture.input.encode("utf-8")
    with Path(a.output).open("wb") as f:
        f.write(struct.pack("<4sII", b"MTL4", len(tokenizer_input), len(ids)))
        f.write(tokenizer_input)
        if ids:
            f.write(struct.pack(f"<{len(ids)}i", *ids))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
