#!/usr/bin/env python
"""FastAPI service that scores protein sequences with low-cycle ESMFold2.

The service loads ``biohub/ESMFold2`` once in bfloat16 and exposes pLDDT/pTM for
the GRPO foldability reward. Folds are serialized through a single GPU worker;
use one uvicorn worker only.

Run (server, repo root):
    python scripts/serving/esmfold2_service.py --host 127.0.0.1 --port 8010
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("esmfold2_service")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
WARMUP_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"


class FoldRequest(BaseModel):
    sequence: str = Field(..., min_length=1)
    num_loops: Optional[int] = Field(default=None, ge=1, le=64)
    num_sampling_steps: Optional[int] = Field(default=None, ge=1, le=400)
    seed: int = 0


class FoldBatchRequest(BaseModel):
    sequences: list[str] = Field(..., min_length=1)
    num_loops: Optional[int] = Field(default=None, ge=1, le=64)
    num_sampling_steps: Optional[int] = Field(default=None, ge=1, le=400)
    seed: int = 0


@dataclass
class ServiceConfig:
    model: str
    device: str
    dtype: str
    esmc_precision: str
    num_loops: int
    num_sampling_steps: int
    max_len: int
    cache_size: int
    api_key: Optional[str]
    warmup: bool


@dataclass
class Runtime:
    executor: ThreadPoolExecutor
    semaphore: asyncio.Semaphore
    cache: OrderedDict = field(default_factory=OrderedDict)
    cache_lock: threading.Lock = field(default_factory=threading.Lock)
    cache_size: int = 512
    autocast_dtype: Optional[torch.dtype] = None
    model: Any = None
    builder: Any = None
    ready: bool = False
    pending: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="biohub/ESMFold2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--esmc-precision", default="bf16", choices=["bf16", "fp8"])
    parser.add_argument("--num-loops", type=int, default=4, help="ESMFold2 recycling loops.")
    parser.add_argument("--num-sampling-steps", type=int, default=20, help="Diffusion sampling steps.")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--api-key", default=None, help="Optional value required in X-API-Key.")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def clean_sequence(sequence: str, max_len: int) -> str:
    sequence = "".join(sequence.split()).upper()
    if not sequence:
        raise ValueError("empty sequence")
    invalid = sorted(set(sequence) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f"non-canonical residues: {''.join(invalid)}")
    if len(sequence) > max_len:
        raise ValueError(f"sequence length {len(sequence)} exceeds max_len={max_len}")
    if len(sequence) < 10:
        raise ValueError("sequence must be at least 10 residues")
    return sequence


def _load_model(cfg: ServiceConfig):
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

    dtype = getattr(torch, cfg.dtype)
    started = time.time()
    LOGGER.info("loading %s (%s, esmc=%s) on %s", cfg.model, cfg.dtype, cfg.esmc_precision, cfg.device)
    model = EsmFold2Model.from_pretrained(
        cfg.model,
        device=cfg.device,
        dtype=dtype,
        esmc_precision=cfg.esmc_precision,
    ).eval()
    LOGGER.info("model loaded in %.1fs", time.time() - started)
    return model, ESMFold2InputBuilder()


def _prepare(rt: Runtime, cfg: ServiceConfig) -> None:
    rt.model, rt.builder = _load_model(cfg)


def _fold(
    rt: Runtime,
    sequence: str,
    num_loops: int,
    num_sampling_steps: int,
    seed: int,
) -> dict[str, Any]:
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    started = time.time()
    prediction_input = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
    autocast = (
        torch.autocast(device_type="cuda", dtype=rt.autocast_dtype)
        if rt.autocast_dtype is not None
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        result = rt.builder.fold(
            rt.model,
            prediction_input,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=1,
            seed=seed,
            lm_dropout=0.0,
        )
    if isinstance(result, list):
        result = result[0]
    plddt = float(torch.as_tensor(result.plddt).float().mean().cpu())
    ptm_value = getattr(result, "ptm", None)
    ptm = None if ptm_value is None else float(ptm_value)
    return {
        "sequence_length": len(sequence),
        "plddt_mean": plddt,
        "plddt_x100": round(plddt * 100.0, 2),
        "ptm": ptm,
        "num_loops": num_loops,
        "num_sampling_steps": num_sampling_steps,
        "seed": seed,
        "elapsed_s": round(time.time() - started, 2),
        "cached": False,
    }


def _fold_cached(
    rt: Runtime,
    sequence: str,
    num_loops: int,
    num_sampling_steps: int,
    seed: int,
) -> dict[str, Any]:
    key = (sequence, num_loops, num_sampling_steps, seed)
    with rt.cache_lock:
        cached = rt.cache.get(key)
        if cached is not None:
            rt.cache.move_to_end(key)
            return {**cached, "cached": True}
    result = _fold(rt, sequence, num_loops, num_sampling_steps, seed)
    with rt.cache_lock:
        rt.cache[key] = result
        while rt.cache_size > 0 and len(rt.cache) > rt.cache_size:
            rt.cache.popitem(last=False)
    return result


def create_app(cfg: ServiceConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rt = Runtime(
            executor=ThreadPoolExecutor(max_workers=1),
            semaphore=asyncio.Semaphore(1),
            cache_size=cfg.cache_size,
            autocast_dtype=None if cfg.dtype == "float32" else getattr(torch, cfg.dtype),
        )
        app.state.rt = rt
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(rt.executor, _prepare, rt, cfg)
        if cfg.warmup:
            try:
                await loop.run_in_executor(
                    rt.executor, _fold, rt, WARMUP_SEQUENCE, 1, 2, 0
                )
                LOGGER.info("warmup fold finished")
            except Exception:  # noqa: BLE001 - warmup must not prevent serving
                LOGGER.exception("warmup fold failed")
        rt.ready = True
        yield
        rt.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Barca ESMFold2 pLDDT service", lifespan=lifespan)
    app.state.cfg = cfg

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if cfg.api_key is not None and x_api_key != cfg.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.get("/health")
    async def health():
        rt: Runtime = app.state.rt
        payload = {
            "ready": rt.ready,
            "model": cfg.model,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "esmc_precision": cfg.esmc_precision,
            "num_loops": cfg.num_loops,
            "num_sampling_steps": cfg.num_sampling_steps,
            "max_len": cfg.max_len,
            "cache_size": len(rt.cache),
            "pending": rt.pending,
        }
        return JSONResponse(payload, status_code=200 if rt.ready else 503)

    async def run_fold(
        rt: Runtime, sequence: str, num_loops: int, num_sampling_steps: int, seed: int
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        rt.pending += 1
        try:
            async with rt.semaphore:
                return await loop.run_in_executor(
                    rt.executor, _fold_cached, rt, sequence, num_loops, num_sampling_steps, seed
                )
        finally:
            rt.pending -= 1

    def resolve_cycles(num_loops: Optional[int], num_sampling_steps: Optional[int]) -> tuple[int, int]:
        return (
            cfg.num_loops if num_loops is None else num_loops,
            cfg.num_sampling_steps if num_sampling_steps is None else num_sampling_steps,
        )

    @app.post("/fold")
    async def fold(request: FoldRequest, _: None = Depends(require_key)):
        rt: Runtime = app.state.rt
        if not rt.ready:
            raise HTTPException(status_code=503, detail="model is still loading")
        try:
            sequence = clean_sequence(request.sequence, cfg.max_len)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        loops, steps = resolve_cycles(request.num_loops, request.num_sampling_steps)
        try:
            return await run_fold(rt, sequence, loops, steps, request.seed)
        except torch.cuda.OutOfMemoryError as exc:
            raise HTTPException(status_code=503, detail="CUDA out of memory") from exc
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("fold failed")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/fold_batch")
    async def fold_batch(request: FoldBatchRequest, _: None = Depends(require_key)):
        rt: Runtime = app.state.rt
        if not rt.ready:
            raise HTTPException(status_code=503, detail="model is still loading")
        loops, steps = resolve_cycles(request.num_loops, request.num_sampling_steps)
        results: list[dict[str, Any]] = []
        for raw in request.sequences:
            try:
                sequence = clean_sequence(raw, cfg.max_len)
            except ValueError as exc:
                results.append({"error": str(exc)})
                continue
            try:
                results.append(await run_fold(rt, sequence, loops, steps, request.seed))
            except torch.cuda.OutOfMemoryError:
                results.append({"error": "CUDA out of memory"})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("batch fold failed")
                results.append({"error": f"{type(exc).__name__}: {exc}"})
        return {"results": results}

    return app


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = ServiceConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        esmc_precision=args.esmc_precision,
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        max_len=args.max_len,
        cache_size=args.cache_size,
        api_key=args.api_key,
        warmup=not args.no_warmup,
    )
    app = create_app(cfg)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level=args.log_level)


if __name__ == "__main__":
    main()
