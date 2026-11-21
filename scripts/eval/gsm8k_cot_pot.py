"""GSM8K few-shot evaluation of a causal LM with chain-of-thought (CoT, Wei et al. 8-shot exemplars) and
program-of-thoughts (PoT: the model writes a Python solution() that is executed). Greedy, fp32, plain text prompts.
Usage: python eval_out/gsm8k_cot_pot.py CKPT OUT.json [N=10] [SEED=0]"""
import json, random, re, subprocess, sys, tempfile, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
ck, outp = sys.argv[1:3]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 10; SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

COT = [("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
       ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
       ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
       ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
       ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
       ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
       ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
       ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.")]
POT = [(COT[0][0], "def solution():\n    trees_initial = 15\n    trees_after = 21\n    trees_added = trees_after - trees_initial\n    result = trees_added\n    return result"),
       (COT[2][0], "def solution():\n    leah_chocolates = 32\n    sister_chocolates = 42\n    total_chocolates = leah_chocolates + sister_chocolates\n    chocolates_eaten = 35\n    result = total_chocolates - chocolates_eaten\n    return result"),
       (COT[5][0], "def solution():\n    computers_initial = 9\n    computers_per_day = 5\n    num_days = 4  # 4 days between monday and thursday\n    computers_added = computers_per_day * num_days\n    result = computers_initial + computers_added\n    return result"),
       (COT[6][0], "def solution():\n    golf_balls_initial = 58\n    golf_balls_lost_tuesday = 23\n    golf_balls_lost_wednesday = 2\n    result = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n    return result"),
       (COT[7][0], "def solution():\n    money_initial = 23\n    bagels = 5\n    bagel_cost = 3\n    money_spent = bagels * bagel_cost\n    result = money_initial - money_spent\n    return result")]
def prompt(shots, q): return "".join(f"Q: {a}\nA: {b}\n\n" for a, b in shots) + f"Q: {q}\nA:"
num = lambda s: float(s.replace(",", ""))
def cot_pred(t):
    m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", t)
    if m: return num(m.group(1))
    n = re.findall(r"-?[\d,]*\.?\d+", t); return num(n[-1]) if n else None
def run_prog(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(code + "\nprint(solution())\n")
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=10)
        return (num(r.stdout.strip().splitlines()[-1]), None) if r.returncode == 0 and r.stdout.strip() else (None, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "no output")
    except Exception as e: return None, type(e).__name__

test = load_dataset("openai/gsm8k", "main", split="test")
idx = random.Random(SEED).sample(range(len(test)), N)
tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True, dtype=torch.float32).cuda().eval()
def gen(p, max_new):
    ids = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, stop_strings=["\n\nQ:", "\nQ:"], tokenizer=tok)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\nQ:")[0].strip()
rows = []
for i in idx:
    q, gold = test[i]["question"], num(test[i]["answer"].split("####")[-1].strip())
    c = gen(prompt(COT, q), 300); cp = cot_pred(c)
    p = gen(prompt(POT, q), 300); pp, err = run_prog(p)
    rows.append(dict(idx=i, question=q, gold=gold, cot=c, cot_pred=cp, cot_ok=cp is not None and abs(cp - gold) < 1e-4,
                     pot=p, pot_pred=pp, pot_err=err, pot_ok=pp is not None and abs(pp - gold) < 1e-4))
    print(f"[{i}] gold {gold:g} | CoT {cp} {'OK' if rows[-1]['cot_ok'] else 'x'} | PoT {pp} {'OK' if rows[-1]['pot_ok'] else 'x'} {err or ''}", flush=True)
print(f"CoT 8-shot: {sum(r['cot_ok'] for r in rows)}/{N}   PoT 5-shot: {sum(r['pot_ok'] for r in rows)}/{N}")
json.dump(rows, open(outp, "w"), ensure_ascii=False, indent=1)
