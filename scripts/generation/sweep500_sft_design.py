"""Generation-config sweep, batched: N val prompts x 12 configs (left-padded batches, sorted by prompt length).
Batched greedy output was verified identical to unbatched on 16 prompts (see sweep40).
Usage (repo root, on the GPU server): python eval_out/sweep500.py CKPT TAG [N]"""
import json, random, re, sys, time, torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
ck, tag = sys.argv[1], sys.argv[2]
BS = 50
tok = AutoTokenizer.from_pretrained("outputs/barca-0.8b-calibration-ft", trust_remote_code=True)
tok.padding_side = "left"
val = load_from_disk("/home/kabidi/workspace/datasets/barca/mol-instruct")["validation"]
N = min(int(sys.argv[3]) if len(sys.argv) > 3 else 500, len(val))
idx = random.Random(789).sample(range(len(val)), N)
model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True, dtype=torch.float32).cuda().eval()
end = tok.convert_tokens_to_ids("<|im_end|>")
AA = set("ACDEFGHIKLMNPQRSTVWY")
km = lambda s, k=5: {s[i:i+k] for i in range(len(s)-k+1)}
P = dict(top_p=0.95, top_k=50)
def S(t, rp): return dict(do_sample=True, temperature=t, repetition_penalty=rp, **P)
def G(rp): return dict(do_sample=False, repetition_penalty=rp)
cfgs = [("greedy rp1.1", G(1.1)), ("greedy rp1.2", G(1.2)), ("greedy rp1.3", G(1.3)),
        ("T0.3 rp1.1", S(0.3, 1.1)), ("T0.3 rp1.2", S(0.3, 1.2)), ("T0.3 rp1.3", S(0.3, 1.3)),
        ("T0.5 rp1.1", S(0.5, 1.1)), ("T0.5 rp1.2", S(0.5, 1.2)), ("T0.6 rp1.2", S(0.6, 1.2)),
        ("T0.8 rp1.0", S(0.8, 1.0)), ("T0.8 rp1.1", S(0.8, 1.1)), ("T0.8 rp1.2", S(0.8, 1.2))]
exs = []
for i in idx:
    ex = val[i]
    m = re.match(r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n(.*?)<\|im_end\|>", ex["text"], re.S)
    ref = m.group(2).split("<protein>")[1].split("</protein>")[0].replace("ζ", "")
    text = tok.apply_chat_template([{"role": "user", "content": m.group(1)}], tokenize=False, add_generation_prompt=True)
    exs.append(dict(idx=i, acc=ex["accession"], prompt=m.group(1), ref=ref, text=text, n=len(tok(text, add_special_tokens=False)["input_ids"])))
exs.sort(key=lambda e: e["n"])
batches = [exs[i:i+BS] for i in range(0, N, BS)]
res = {}
print(f"{N} prompts, {len(batches)} batches of <= {BS}", flush=True)
print(f"{'config':<14}{'closed':>9}{'canon':>8}{'len/ref':>9}{'uniq5':>7}{'time':>7}", flush=True)
for name, kw in cfgs:
    t = time.time(); rows = []
    for b in batches:
        enc = tok([e["text"] for e in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.manual_seed(0)
        out = model.generate(**enc, max_new_tokens=1100, eos_token_id=end, **kw)
        for e, o in zip(b, out):
            gen = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=False)
            s = gen.split("<protein>")[1].split("</protein>")[0].replace("ζ", "") if "<protein>" in gen else ""
            rows.append(dict(idx=e["idx"], acc=e["acc"], ref_len=len(e["ref"]), gen_len=len(s), gen_seq=s, ref_seq=e["ref"], closed="</protein>" in gen,
                             canon=bool(s) and set(s) <= AA, uniq5=len(km(s)) / max(1, len(s) - 4),
                             maxfrac=(max(s.count(a) for a in set(s)) / len(s)) if s else 1.0,
                             refcov=len(km(s) & km(e["ref"])) / max(1, len(km(e["ref"])))))
    res[name] = rows
    c = [r for r in rows if r["closed"]]
    ratio = sum(r["gen_len"] / r["ref_len"] for r in c) / len(c) if c else float("nan")
    print(f"{name:<14}{len(c):>5}/{N}{sum(r['canon'] for r in rows):>5}/{N}{ratio:>8.2f}{sum(r['uniq5'] for r in rows)/N:>7.2f}{time.time()-t:>6.0f}s", flush=True)
    json.dump(dict(prompts={e["idx"]: e["prompt"] for e in exs}, res=res), open(f"eval_out/sweep500_{tag}.json", "w"))
print("DONE")
