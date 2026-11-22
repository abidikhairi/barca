#!/usr/bin/env python
"""Smoke-test the ESMFold2 service on a few sequences and log to W&B.

The script waits for the service to become ready, folds each sequence through
``POST /fold``, prints a table, and logs per-fold metrics plus a summary table
to Weights & Biases. It is intended to run on the GPU server next to the
service.

Usage:
    python scripts/serving/smoke_fold_wandb.py --input /tmp/smoke10.json \
        --url http://127.0.0.1:8010 --project Barca
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="FASTA or JSON list of records.")
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-loops", type=int, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=1800.0, help="HTTP timeout per fold (s).")
    parser.add_argument("--wait", type=float, default=1200.0, help="Seconds to wait for /health.")
    parser.add_argument("--project", default="Barca")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [{"header": key, "sequence": value} for key, value in payload.items()]
        records = []
        for index, item in enumerate(payload):
            if isinstance(item, str):
                records.append({"header": f"seq{index}", "sequence": item})
            elif isinstance(item, dict):
                sequence = item.get("sequence") or item.get("seq")
                if not sequence:
                    raise ValueError(f"record {index} has no sequence")
                records.append({"header": str(item.get("header", item.get("id", f"seq{index}"))), "sequence": sequence})
            else:
                raise ValueError(f"unsupported record {type(item).__name__} at {index}")
        return records

    records = []
    header = None
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append({"header": header, "sequence": "".join(chunks)})
            header = line[1:].strip() or f"seq{len(records)}"
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append({"header": header, "sequence": "".join(chunks)})
    return records


def wait_for_ready(client: httpx.Client, url: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            response = client.get(f"{url}/health")
            payload = response.json()
            if response.status_code == 200 and payload.get("ready"):
                return payload
            last_error = f"status={response.status_code} ready={payload.get('ready')}"
        except Exception as exc:  # noqa: BLE001 - service may still be loading
            last_error = f"{type(exc).__name__}: {exc}"
        print(f"waiting for service: {last_error}", flush=True)
        time.sleep(10)
    raise TimeoutError(f"service not ready within {timeout:.0f}s: {last_error}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    ok = [row for row in rows if row.get("status") == "ok"]
    plddt = [row["plddt_mean"] for row in ok]
    latency = [row["elapsed_s"] for row in ok]
    return {
        "n": float(len(rows)),
        "n_ok": float(len(ok)),
        "n_error": float(len(rows) - len(ok)),
        "plddt_mean": statistics.mean(plddt) if plddt else float("nan"),
        "plddt_median": statistics.median(plddt) if plddt else float("nan"),
        "plddt_min": min(plddt) if plddt else float("nan"),
        "plddt_max": max(plddt) if plddt else float("nan"),
        "elapsed_mean_s": statistics.mean(latency) if latency else float("nan"),
        "elapsed_median_s": statistics.median(latency) if latency else float("nan"),
    }


def main() -> None:
    args = parse_args()
    records = read_records(args.input)[: args.limit]
    if not records:
        raise SystemExit(f"no records in {args.input}")
    print(f"smoke-testing {len(records)} sequences from {args.input}", flush=True)

    with httpx.Client(timeout=args.timeout) as client:
        health = wait_for_ready(client, args.url, args.wait)
        print("service ready:", json.dumps(health), flush=True)
        wandb = None
        run = None
        if not args.no_wandb:
            import wandb as wandb_module

            wandb = wandb_module
            run = wandb.init(
                project=args.project,
                entity=args.entity,
                name=args.name or f"esmfold2-service-smoke-{int(time.time())}",
                tags=["esmfold2", "smoke", "foldability"],
                config={
                    "service": health,
                    "input": str(args.input),
                    "n_sequences": len(records),
                    "seed": args.seed,
                    "num_loops": args.num_loops,
                    "num_sampling_steps": args.num_sampling_steps,
                },
            )
            print(f"wandb run: {run.url}", flush=True)

        rows: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            header = record["header"]
            started = time.time()
            try:
                payload = {
                    "sequence": record["sequence"],
                    "seed": args.seed,
                }
                if args.num_loops is not None:
                    payload["num_loops"] = args.num_loops
                if args.num_sampling_steps is not None:
                    payload["num_sampling_steps"] = args.num_sampling_steps
                response = client.post(f"{args.url}/fold", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    row = {
                        "header": header,
                        "length": result["sequence_length"],
                        "status": "ok",
                        "plddt_mean": result["plddt_mean"],
                        "plddt_x100": result["plddt_x100"],
                        "ptm": result.get("ptm"),
                        "elapsed_s": result["elapsed_s"],
                        "num_loops": result["num_loops"],
                        "num_sampling_steps": result["num_sampling_steps"],
                        "cached": result.get("cached", False),
                    }
                else:
                    row = {
                        "header": header,
                        "length": len(record["sequence"]),
                        "status": "error",
                        "elapsed_s": round(time.time() - started, 2),
                        "error": f"HTTP {response.status_code}: {response.text[:300]}",
                    }
            except Exception as exc:  # noqa: BLE001 - report and keep going
                row = {
                    "header": header,
                    "length": len(record["sequence"]),
                    "status": "error",
                    "elapsed_s": round(time.time() - started, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            if row["status"] == "ok":
                print(
                    f"[{index}/{len(records)}] {header} len {row['length']} "
                    f"pLDDT {row['plddt_x100']:.2f} pTM {row['ptm']} "
                    f"{row['elapsed_s']:.2f}s cached={row['cached']}",
                    flush=True,
                )
                if wandb is not None:
                    wandb.log(
                        {
                            "fold/plddt": row["plddt_mean"],
                            "fold/plddt_x100": row["plddt_x100"],
                            "fold/ptm": row["ptm"],
                            "fold/length": row["length"],
                            "fold/elapsed_s": row["elapsed_s"],
                        },
                        step=index,
                    )
            else:
                print(f"[{index}/{len(records)}] {header} ERROR {row['error']}", flush=True)
                if wandb is not None:
                    wandb.log({"fold/error": 1.0}, step=index)

        summary = summarize(rows)
        print("summary:", json.dumps(summary, indent=2), flush=True)
        if wandb is not None:
            table = wandb.Table(
                columns=[
                    "header",
                    "length",
                    "status",
                    "plddt_mean",
                    "plddt_x100",
                    "ptm",
                    "elapsed_s",
                    "num_loops",
                    "num_sampling_steps",
                    "cached",
                    "error",
                ]
            )
            for row in rows:
                table.add_data(
                    row["header"],
                    row["length"],
                    row["status"],
                    row.get("plddt_mean"),
                    row.get("plddt_x100"),
                    row.get("ptm"),
                    row.get("elapsed_s"),
                    row.get("num_loops"),
                    row.get("num_sampling_steps"),
                    row.get("cached"),
                    row.get("error", ""),
                )
            wandb.log({"folds": table, **{f"summary/{key}": value for key, value in summary.items()}})
            print(f"wandb run: {run.url}", flush=True)
            wandb.finish()

    errors = [row for row in rows if row["status"] != "ok"]
    if errors:
        raise SystemExit(f"{len(errors)} of {len(rows)} folds failed")


if __name__ == "__main__":
    main()
