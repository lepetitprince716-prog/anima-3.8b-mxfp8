#!/usr/bin/env python3
"""Quantize the Anima 3.8B v1.1 bundle to MXFP8 for ComfyUI, then verify the result.

Run it with the Python environment of a ComfyUI install whose comfy_kitchen supports MXFP8
(the published file was made with ComfyUI 0.42.1 and comfy-kitchen 0.2.39):

    python bake_mxfp8.py --comfy <ComfyUI directory> SRC DST [--require-native]

Quantized to MXFP8: the self-attention, cross-attention and MLP projection weights of the
52 DiT blocks, 520 tensors in all. Everything else is copied unchanged, including
llm_adapter and the bundled Semantic Connector v2 (net.anima_v2_connector.*). The upstream
metadata is kept, so the anima.3-8B-v2 loader still lists the file. The MXFP8 block scales are
stored as uint8 bytes, the form ComfyUI's mmap loader reads.

The verification reloads the file the way the anima.3-8B-v2 loader does and checks the
quantized layer set. With --require-native it also fails when ComfyUI would dequantize the
layers on this GPU instead of running native MXFP8 matmuls (that needs an RTX 50-series card).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

LAYER_RE = re.compile(
    r"net\.blocks\.(\d+)\.((?:self_attn|cross_attn)\.(?:q|k|v|output)_proj|mlp\.layer[12])\.weight"
)
CONNECTOR_PREFIX = "net.anima_v2_connector."
EXPECTED_BLOCKS = 52
EXPECTED_LAYERS = EXPECTED_BLOCKS * 10
FORMAT = "mxfp8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    return state, metadata


def versions(comfy_dir: Path) -> dict[str, str]:
    found = {}
    try:
        import comfyui_version

        found["comfyui"] = comfyui_version.__version__
    except ImportError:
        found["comfyui"] = "unknown"
    try:
        found["comfy_kitchen"] = importlib_metadata.version("comfy-kitchen")
    except importlib_metadata.PackageNotFoundError:
        found["comfy_kitchen"] = "unknown"
    return found


def quantize(state: dict, device: str) -> list[str]:
    import torch

    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor

    if FORMAT not in QUANT_ALGOS:
        raise SystemExit(f"this ComfyUI has no {FORMAT} support (comfy_kitchen too old?); have {sorted(QUANT_ALGOS)}")
    layout = QUANT_ALGOS[FORMAT]["comfy_tensor_layout"]
    group = int(QUANT_ALGOS[FORMAT].get("group_size", 32))

    layers = sorted(key[: -len(".weight")] for key in state if LAYER_RE.fullmatch(key))
    blocks = {int(LAYER_RE.fullmatch(f"{layer}.weight").group(1)) for layer in layers}
    if len(layers) != EXPECTED_LAYERS or blocks != set(range(EXPECTED_BLOCKS)):
        raise SystemExit(
            f"expected {EXPECTED_LAYERS} layers over {EXPECTED_BLOCKS} blocks, "
            f"found {len(layers)} over {len(blocks)}"
        )

    marker = list(json.dumps({"format": FORMAT}).encode("utf-8"))
    started = time.perf_counter()
    for index, layer in enumerate(layers, 1):
        weight = state.pop(f"{layer}.weight")
        if weight.ndim != 2 or weight.shape[1] % group:
            raise SystemExit(f"{layer}: {FORMAT} needs a 2-D weight with K % {group} == 0, got {tuple(weight.shape)}")
        quantized = QuantizedTensor.from_float(weight.to(device, torch.bfloat16), layout)
        written = quantized.state_dict(f"{layer}.weight")
        for key, value in written.items():
            # ComfyUI's mmap loader has no F8_E8M0 entry; it reads the block scales as uint8
            # and views them as float8_e8m0fnu, so store the same bytes as uint8.
            if value.dtype == getattr(torch, "float8_e8m0fnu", None):
                value = value.view(torch.uint8)
            written[key] = value
            state[key] = value.cpu().contiguous()
        state[f"{layer}.comfy_quant"] = torch.tensor(marker, dtype=torch.uint8)
        if index == 1:
            for key, value in written.items():
                print(f"LAYOUT {key} {value.dtype} {tuple(value.shape)}", flush=True)
        if index % 104 == 0 or index == len(layers):
            print(f"quantized {index}/{len(layers)}", flush=True)
        del weight, quantized, written
    print(f"QUANTIZE_SECONDS {time.perf_counter() - started:.1f}", flush=True)
    return layers


def save(state: dict, metadata: dict, dst: Path) -> None:
    from safetensors.torch import save_file

    partial = dst.with_name(dst.stem + ".partial")
    try:
        save_file(state, str(partial), metadata=metadata)
        os.replace(partial, dst)
    finally:
        if partial.exists():
            partial.unlink()


def verify(dst: Path, layers: list[str], connector_count: int, require_native: bool) -> None:
    import torch

    import comfy.sd
    import comfy.utils
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor
    from safetensors import safe_open

    # The anima.3-8B-v2 loader reads the file with comfy.utils.load_torch_file, which uses a
    # fixed dtype table when DynamicVRAM is on; every stored dtype has to be in it.
    known = getattr(comfy.utils, "_TYPES", None)
    if known is not None:
        with safe_open(str(dst), framework="pt", device="cpu") as handle:
            stored = {handle.get_slice(key).get_dtype() for key in handle.keys()}
        unknown = sorted(stored - set(known))
        if unknown:
            raise SystemExit(f"ComfyUI's safetensors loader cannot read dtypes {unknown}")

    state, metadata = comfy.utils.load_torch_file(str(dst), return_metadata=True)
    metadata = metadata or {}
    connector = [key for key in state if key.startswith(CONNECTOR_PREFIX)]
    if len(connector) != connector_count:
        raise SystemExit(f"connector tensors: wrote {connector_count}, read back {len(connector)}")
    for key in connector:
        state.pop(key)

    patcher = comfy.sd.load_diffusion_model_state_dict(state, model_options={}, metadata=metadata)
    if patcher is None:
        raise SystemExit("ComfyUI could not load the quantized DiT")
    model = patcher.model.diffusion_model
    quant = {
        name: module
        for name, module in model.named_modules()
        if isinstance(getattr(module, "weight", None), QuantizedTensor)
    }
    expected = sorted(layer[len("net."):] for layer in layers)
    if sorted(quant) != expected:
        raise SystemExit(f"loaded {len(quant)} quantized layers, expected {len(expected)}")

    layout = QUANT_ALGOS[FORMAT]["comfy_tensor_layout"]
    wrong = [n for n, m in quant.items() if m.quant_format != FORMAT or m.layout_type != layout]
    if wrong:
        raise SystemExit(f"{len(wrong)} layers carry the wrong format, e.g. {wrong[:3]}")
    # A missing flag means the ComfyUI API changed and the fallback check below would pass vacuously.
    missing = [n for n, m in quant.items() if not hasattr(m, "_full_precision_mm")]
    if missing:
        raise SystemExit(f"{len(missing)} quantized layers have no _full_precision_mm flag; cannot check fallback")
    emulated = [n for n, m in quant.items() if m._full_precision_mm]
    if emulated:
        message = f"{len(emulated)} layers fall back to dequantize on this GPU (native MXFP8 needs Blackwell)"
        if require_native:
            raise SystemExit(message)
        print(f"WARN {message}", flush=True)

    if len(model.blocks) != EXPECTED_BLOCKS:
        raise SystemExit(f"model has {len(model.blocks)} blocks, expected {EXPECTED_BLOCKS}")
    adapter = model.llm_adapter
    if adapter.embed.weight.dtype != torch.bfloat16 or len(adapter.blocks) != 6:
        raise SystemExit(f"llm_adapter changed: embed {adapter.embed.weight.dtype}, {len(adapter.blocks)} blocks")
    print(
        f"VERIFY_OK quantized={len(quant)} native={len(quant) - len(emulated)} "
        f"blocks={len(model.blocks)} connector_tensors={len(connector)} llm_adapter_embed={adapter.embed.weight.dtype}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="Anima-3.8B-v1.1.safetensors")
    parser.add_argument("dst", type=Path, help="output .safetensors")
    parser.add_argument("--comfy", type=Path, required=True, help="ComfyUI directory (the one with main.py)")
    parser.add_argument("--device", default="cuda", help="device for the quantization kernels")
    parser.add_argument("--require-native", action="store_true", help="fail if MXFP8 matmuls would be emulated")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if dst.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {dst}")
    sys.path.insert(0, str(args.comfy.resolve()))
    os.chdir(args.comfy.resolve())

    print(f"hash {src.name}", flush=True)
    source_sha256 = sha256_file(src)
    print(f"SOURCE_SHA256 {source_sha256}", flush=True)
    state, metadata = load(src)
    connector_count = sum(1 for key in state if key.startswith(CONNECTOR_PREFIX))
    print(f"SOURCE tensors={len(state)} connector={connector_count} metadata_keys={len(metadata)}", flush=True)

    import comfy.quant_ops  # noqa: F401  registers ComfyUI's layout classes

    layers = quantize(state, args.device)
    metadata["anima_mxfp8_bake"] = json.dumps(
        {
            "source": src.name,
            "source_sha256": source_sha256,
            "format": FORMAT,
            "quantized_layers": len(layers),
            "quantized": "net.blocks.N.{self_attn,cross_attn}.{q,k,v,output}_proj, net.blocks.N.mlp.layer1/2",
            **versions(args.comfy),
        }
    )
    save(state, metadata, dst)
    del state
    print(f"BAKED {dst.name} bytes={dst.stat().st_size}", flush=True)

    verify(dst, layers, connector_count, args.require_native)
    print(f"OUTPUT_SHA256 {sha256_file(dst)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
