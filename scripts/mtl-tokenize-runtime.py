#!/usr/bin/env python3
import argparse
import ast
import importlib.util
import json
import struct
import sys
import traceback
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
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        raise RuntimeError(f"{name} not found in {path}")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class CaptureTokenizer:
    def __init__(self, inner):
        self.inner = inner
        self.input = None

    def encode(self, text, *args, **kwargs):
        self.input = text
        return self.inner.encode(text, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class Runtime:
    def __init__(self, args):
        self.language = args.language
        self.punc = load_function(Path(args.tts_source).resolve(), "punc_norm")
        module = load_module(Path(args.source).resolve())
        cangjie_data = json.loads(Path(args.cangjie).read_text(encoding="utf-8"))

        def load_cangjie(instance, model_dir=None):
            instance.word2cj = {}
            instance.cj2word = {}
            for entry in cangjie_data:
                word, code = entry.split("\t")[:2]
                instance.word2cj[word] = code
                instance.cj2word.setdefault(code, []).append(word)

        module.ChineseCangjieConverter._load_cangjie_mapping = load_cangjie
        if self.language == "he":
            from dicta_onnx import Dicta
            module._dicta = Dicta(str(Path(args.dicta_model).resolve()))
        elif self.language == "ja":
            import pykakasi
            module._kakasi = pykakasi.kakasi()
        elif self.language == "ru":
            from russian_text_stresser.text_stresser import RussianTextStresser
            module._russian_stresser = RussianTextStresser()
        elif self.language == "zh":
            import spacy_pkuseg  # noqa: F401
        self.tokenizer = module.MTLTokenizer(str(Path(args.tokenizer).resolve()))
        self.capture = CaptureTokenizer(self.tokenizer.tokenizer)
        self.tokenizer.tokenizer = self.capture

    def punctuation(self, text: str) -> str:
        return self.punc(text)

    def tokenize(self, text: str):
        self.capture.input = None
        ids = self.tokenizer.encode(text, language_id=self.language)
        if self.capture.input is None:
            raise RuntimeError("official tokenizer encode did not execute")
        return self.capture.input, ids


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise EOFError
        data.extend(block)
    return bytes(data)


def response_payload(runtime: Runtime, mode: bytes, text: str) -> bytes:
    if mode == b"R":
        return b"ready"
    if mode == b"P":
        return runtime.punctuation(text).encode("utf-8")
    if mode == b"T":
        tokenizer_input, ids = runtime.tokenize(text)
        encoded = tokenizer_input.encode("utf-8")
        return struct.pack("<I", len(encoded)) + encoded + struct.pack("<I", len(ids)) + (struct.pack(f"<{len(ids)}i", *ids) if ids else b"")
    raise RuntimeError("unknown tokenizer request mode")


def serve(runtime: Runtime):
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        mode = inp.read(1)
        if not mode:
            return 0
        try:
            size = struct.unpack("<I", read_exact(inp, 4))[0]
            text = read_exact(inp, size).decode("utf-8")
            payload = response_payload(runtime, mode, text)
            out.write(struct.pack("<II", 0, len(payload)))
            out.write(payload)
        except BaseException:
            payload = traceback.format_exc().encode("utf-8", errors="replace")
            out.write(struct.pack("<II", 1, len(payload)))
            out.write(payload)
        out.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--tts-source", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--cangjie", required=True)
    p.add_argument("--dicta-model", required=True)
    p.add_argument("--language", required=True)
    args = p.parse_args()
    return serve(Runtime(args))


if __name__ == "__main__":
    raise SystemExit(main())
