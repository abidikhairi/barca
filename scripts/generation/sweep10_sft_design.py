import json, random, re, sys, time, torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
ck, tag = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained("outputs/barca-0.8b-calibration-ft", trust_remote_code=True)
val = load_from_disk("/home/kabidi/workspace/datasets/barca/mol-instruct")["validation"]
idx = random.Random(123).sample(range(len(val)), 10)   # same 10 rows as gen10.py
model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True, dtype=torch.float32).cuda().eval()
end = tok.convert_tokens_to_ids("<|im_end|>")
AA = set("ACDEFGHIKLMNPQRSTVWY")
km = lambda s, k=5: {s[i:i+k] for i in range(len(s)-k+1)}
P = dict(top_p=0.95, top_k=50)
cfgs = [("greedy rp1.0", dict(do_sample=False)),
        ("greedy rp1.2", dict(do_sample=False, repetition_penalty=1.2)),
        ("T0.3 rp1.0", dict(do_sample=True, temperature=0.3, **P)),
        ("T0.3 rp1.2", dict(do_sample=True, temperature=0.3, repetition_penalty=1.2, **P)),
        ("T0.5 rp1.0", dict(do_sample=True, temperature=0.5, **P)),
        ("T0.6 rp1.2", dict(do_sample=True, temperature=0.6, repetition_penalty=1.2, **P)),
        ("T0.8 rp1.0", dict(do_sample=True, temperature=0.8, **P)),
        ("T1.0 k0 p1.0", dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0))]
exs = []
for i in idx:
    ex = val[i]
    m = re.match(r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n(.*?)<\|im_end\|>", ex["text"], re.S)
    ref_seq = m.group(2).split("<protein>")[1].split("</protein>")[0].replace("ζ", "")
    ids = tok(tok.apply_chat_template([{"role": "user", "content": m.group(1)}], tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).to("cuda")
    exs.append((i, ex["accession"], ref_seq, ids))
res = {}
print(f"{'config':<14}{'closed':>7}{'canon':>7}{'len/ref(closed)':>17}{'uniq5':>7}{'refcov':>8}{'time':>6}", flush=True)
for name, kw in cfgs:
    t = time.time(); rows = []
    for i, acc, ref, ids in exs:
        torch.manual_seed(0)
        out = model.generate(**ids, max_new_tokens=1100, eos_token_id=end, **kw)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
        s = gen.split("<protein>")[1].split("</protein>")[0].replace("ζ", "") if "<protein>" in gen else ""
        rows.append(dict(idx=i, acc=acc, ref_len=len(ref), gen_len=len(s), gen_seq=s, closed="</protein>" in gen, canon=set(s) <= AA,
                         uniq5=len(km(s)) / max(1, len(s) - 4), refcov=len(km(s) & km(ref)) / max(1, len(km(ref)))))
    res[name] = rows
    c = [r for r in rows if r["closed"]]
    ratio = sum(r["gen_len"] / r["ref_len"] for r in c) / len(c) if c else float("nan")
    print(f"{name:<14}{len(c):>5}/10{sum(r['canon'] for r in rows):>5}/10{ratio:>17.2f}{sum(r['uniq5'] for r in rows)/10:>7.2f}{sum(r['refcov'] for r in rows)/10:>8.3f}{time.time()-t:>5.0f}s", flush=True)
    json.dump(res, open(f"eval_out/sweep10_{tag}.json", "w"))
print("DONE")
