#!/usr/bin/env python3
import argparse
import ast
import importlib.util
import json
import struct
import sys
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("chatterbox_official_tokenizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


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

    def punctuation(self, text: str) -> str:
        return self.punc(text)

    def tokenize(self, text: str):
        return self.tokenizer.encode(text, language_id=self.language)


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
        ids = runtime.tokenize(text)
        return struct.pack(f"<I{len(ids)}i", len(ids), *ids)
    raise RuntimeError("unknown tokenizer request mode")


def serve(runtime: Runtime):
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        mode = inp.read(1)
        if not mode:
            return 0
        size = struct.unpack("<I", read_exact(inp, 4))[0]
        text = read_exact(inp, size).decode("utf-8")
        payload = response_payload(runtime, mode, text)
        out.write(struct.pack("<II", 0, len(payload)))
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
