"""Wider generation-config sweep for the SFT-design model: 40 val prompts x 12 configs.
Usage (repo root, on the GPU server): python eval_out/sweep40.py CKPT TAG
Saves every generation (eval_out/sweep40_<TAG>.json) so configs can be scored/folded later."""
import json, random, re, sys, time, torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
ck, tag = sys.argv[1], sys.argv[2]
N = 40
tok = AutoTokenizer.from_pretrained("outputs/barca-0.8b-calibration-ft", trust_remote_code=True)
val = load_from_disk("/home/kabidi/workspace/datasets/barca/mol-instruct")["validation"]
idx = random.Random(456).sample(range(len(val)), N)
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
    ids = tok(tok.apply_chat_template([{"role": "user", "content": m.group(1)}], tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).to("cuda")
    exs.append((i, ex["accession"], m.group(1), ref, ids))
res = {}
print(f"{'config':<14}{'closed':>8}{'canon':>8}{'len/ref':>9}{'uniq5':>7}{'time':>7}", flush=True)
for name, kw in cfgs:
    t = time.time(); rows = []
    for i, acc, prompt, ref, ids in exs:
        torch.manual_seed(0)
        out = model.generate(**ids, max_new_tokens=1100, eos_token_id=end, **kw)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
        s = gen.split("<protein>")[1].split("</protein>")[0].replace("ζ", "") if "<protein>" in gen else ""
        rows.append(dict(idx=i, acc=acc, ref_len=len(ref), gen_len=len(s), gen_seq=s, ref_seq=ref, closed="</protein>" in gen,
                         canon=bool(s) and set(s) <= AA, uniq5=len(km(s)) / max(1, len(s) - 4),
                         maxfrac=(max(s.count(a) for a in set(s)) / len(s)) if s else 1.0,
                         refcov=len(km(s) & km(ref)) / max(1, len(km(ref)))))
    res[name] = rows
    c = [r for r in rows if r["closed"]]
    ratio = sum(r["gen_len"] / r["ref_len"] for r in c) / len(c) if c else float("nan")
    print(f"{name:<14}{len(c):>6}/{N}{sum(r['canon'] for r in rows):>6}/{N}{ratio:>9.2f}{sum(r['uniq5'] for r in rows)/N:>7.2f}{time.time()-t:>6.0f}s", flush=True)
    json.dump(dict(prompts={r["idx"]: p for r, (_, _, p, _, _) in zip(rows, exs)}, res=res), open(f"eval_out/sweep40_{tag}.json", "w"))
print("DONE")
