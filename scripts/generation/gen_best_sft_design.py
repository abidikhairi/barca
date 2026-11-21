"""Sample N validation instructions (reference <= MAXREF aa so ESMFold can fold them) and generate with the
chosen config (configs/generation/sft_design_best.yaml). Usage (server, repo root): python eval_out/gen_best.py CKPT OUT.json
"""

import json, random, re, sys, torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

ck, outp = sys.argv[1:3]
N, MAXREF = 10, 400
tok = AutoTokenizer.from_pretrained(
    "outputs/barca-0.8b-calibration-ft", trust_remote_code=True
)
tok.padding_side = "left"
val = load_from_disk("/home/kabidi/workspace/datasets/barca/mol-instruct")["validation"]
pat = re.compile(
    r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n(.*?)<\|im_end\|>",
    re.S,
)
elig = []
for i in range(len(val)):
    m = pat.match(val[i]["text"])
    ref = m.group(2).split("<protein>")[1].split("</protein>")[0].replace("ζ", "")
    if len(ref) <= MAXREF:
        elig.append((i, m.group(1), ref))
print(len(elig), "of", len(val), "rows have ref <=", MAXREF, flush=True)
rows = random.Random(2026).sample(elig, N)
model = (
    AutoModelForCausalLM.from_pretrained(
        ck, trust_remote_code=True, dtype=torch.float32
    )
    .cuda()
    .eval()
)
end = tok.convert_tokens_to_ids("<|im_end|>")
texts = [
    tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
    )
    for _, p, _ in rows
]
enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
torch.manual_seed(0)
out = model.generate(
    **enc,
    max_new_tokens=1100,
    eos_token_id=end,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.2,
)
AA = set("ACDEFGHIKLMNPQRSTVWY")
res = []
for (i, p, ref), o in zip(rows, out):
    g = tok.decode(o[enc["input_ids"].shape[1] :], skip_special_tokens=False)
    s = (
        g.split("<protein>")[1].split("</protein>")[0].replace("ζ", "")
        if "<protein>" in g
        else ""
    )
    res.append(
        dict(
            idx=i,
            acc=val[i]["accession"],
            prompt=p,
            ref_seq=ref,
            gen_seq=s,
            closed="</protein>" in g,
            canon=bool(s) and set(s) <= AA,
        )
    )
    print(
        i,
        val[i]["accession"],
        "ref",
        len(ref),
        "gen",
        len(s),
        "closed",
        res[-1]["closed"],
        "canon",
        res[-1]["canon"],
        flush=True,
    )
json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1)
