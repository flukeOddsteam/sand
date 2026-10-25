"""
สมองของแซนด์ — Transformer ที่เขียนขึ้นเองทุกสูตร
PyTorch ใช้แค่เป็นเครื่องคิดเลขเมทริกซ์ (และคำนวณอนุพันธ์ตอนเรียน) ส่วนโครงสร้างสมองเขียนเองทั้งหมด:

  x₀ = E[ตัวอักษร] + P[ตำแหน่ง]                       (embedding)
  ทุกชั้น:
    h  = LayerNorm(x)       = (x − μ) / √(σ² + ε) · γ + β
    Q, K, V = h·W_q, h·W_k, h·W_v
    A  = softmax(Q·Kᵀ / √d_k + mask)                    (attention: มองย้อนไปอ่านตัวก่อนหน้า)
    x  = x + (A·V)·W_o
    x  = x + GELU(LayerNorm(x)·W₁)·W₂                   (คิดต่อ)
  logits z = LayerNorm(x)·Eᵀ
  p(ตัวถัดไป) = softmax(z / T)
"""
import math
import unicodedata

import torch
import torch.nn.functional as Fn

SIZES = {
    "small":  {"d": 128, "layers": 4, "heads": 4, "ctx": 128, "batch": 32, "label": "เล็ก"},
    "medium": {"d": 256, "layers": 6, "heads": 8, "ctx": 256, "batch": 24, "label": "กลาง"},
    "large":  {"d": 384, "layers": 8, "heads": 8, "ctx": 256, "batch": 16, "label": "ใหญ่"},
}

USER_TAG = "ผู้ใช้: "
SAND_TAG = "แซนด์: "


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"      # GPU ของ Mac ชิป M
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------- ตัวอักษร (tokenizer)
class Tokenizer:
    """แซนด์อ่านทีละตัวอักษร ภาษาไทยไม่มีเว้นวรรคคั่นคำ จึงเหมาะกับการอ่านทีละตัว"""

    def __init__(self, chars=None):
        if chars is None:
            chars = ["\n", " "] + [chr(i) for i in range(33, 127)]
            chars += [chr(i) for i in range(0x0E01, 0x0E5C) if unicodedata.name(chr(i), None)]
            chars += list("“”‘’…–—•")
        self.itos = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.unk = self.stoi["?"]

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi.get(c, self.unk) for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def add_chars(self, chars):
        """เพิ่มตัวอักษรใหม่ (เช่น ภาษาญี่ปุ่น จีน เกาหลี) ต่อท้าย ไม่กระทบ id เดิม"""
        new = [c for c in chars if c not in self.stoi]
        for c in new:
            self.stoi[c] = len(self.itos)
            self.itos.append(c)
        return new


# ---------------------------------------------------------------- สมอง
class Brain(torch.nn.Module):
    def __init__(self, vocab, d=256, layers=6, heads=8, ctx=256, dropout=0.1, **_):
        super().__init__()
        assert d % heads == 0
        self.cfg = {"vocab": vocab, "d": d, "layers": layers, "heads": heads, "ctx": ctx, "dropout": dropout}
        g = lambda *s, std=0.02: torch.nn.Parameter(torch.randn(*s) * std)
        zeros = lambda *s: torch.nn.Parameter(torch.zeros(*s))
        ones = lambda *s: torch.nn.Parameter(torch.ones(*s))
        deep = 0.02 / math.sqrt(2 * layers)   # ชั้นลึกเริ่มเบาๆ ให้ฝึกได้เสถียร

        self.E = g(vocab, d)     # ความหมายของแต่ละตัวอักษร
        self.P = g(ctx, d)       # ความหมายของตำแหน่ง
        self.blocks = torch.nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(torch.nn.ParameterDict({
                "g1": ones(d), "b1": zeros(d),
                "Wqkv": g(d, 3 * d), "bqkv": zeros(3 * d),
                "Wo": g(d, d, std=deep), "bo": zeros(d),
                "g2": ones(d), "b2": zeros(d),
                "W1": g(d, 4 * d), "c1": zeros(4 * d),
                "W2": g(4 * d, d, std=deep), "c2": zeros(d),
            }))
        self.gf, self.bf = ones(d), zeros(d)
        self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx, dtype=torch.bool)), persistent=False)

    def expand_vocab(self, n_new):
        """ขยายสมองให้รู้จักตัวอักษรใหม่ n_new ตัว: ต่อแถวใหม่ใน E โดยน้ำหนักเดิมไม่เปลี่ยน"""
        if n_new <= 0:
            return
        old = self.E.data
        extra = torch.randn(n_new, old.shape[1], device=old.device, dtype=old.dtype) * 0.02
        self.E = torch.nn.Parameter(torch.cat([old, extra], 0))
        self.cfg["vocab"] += n_new

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def layernorm(x, gamma, beta, eps=1e-5):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + eps) * gamma + beta

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, idx, targets=None, trace=False):
        B, T = idx.shape
        d, H = self.cfg["d"], self.cfg["heads"]
        dk = d // H
        p = self.cfg["dropout"] if self.training else 0.0
        x = Fn.dropout(self.E[idx] + self.P[:T], p, self.training)
        tr = [] if trace else None
        for b in self.blocks:
            h = self.layernorm(x, b["g1"], b["b1"])
            q, k, v = (h @ b["Wqkv"] + b["bqkv"]).split(d, dim=-1)
            q, k, v = (t.view(B, T, H, dk).transpose(1, 2) for t in (q, k, v))
            dots = q @ k.transpose(-2, -1)                       # q·k
            scores = dots / math.sqrt(dk)                        # ÷ √d_k
            scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))   # ห้ามแอบดูอนาคต
            A = torch.softmax(scores, dim=-1)
            y = (Fn.dropout(A, p, self.training) @ v).transpose(1, 2).reshape(B, T, d)
            x = x + Fn.dropout(y @ b["Wo"] + b["bo"], p, self.training)
            h2 = self.layernorm(x, b["g2"], b["b2"])
            x = x + Fn.dropout(self.gelu(h2 @ b["W1"] + b["c1"]) @ b["W2"] + b["c2"], p, self.training)
            if trace:
                tr.append({"A": A[0, :, -1, :], "dots": dots[0, :, -1, :], "norm": x[0, -1].norm().item()})
        x = self.layernorm(x, self.gf, self.bf)
        logits = x @ self.E.T
        loss = None
        if targets is not None:
            loss = Fn.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, tr


def _scaled_sum(v):
    """Σ eᵛ แบบไม่ล้น (ถ้าใหญ่เกินจะได้ inf)"""
    m = float(v.max())
    try:
        return float(torch.exp(v - m).sum()) * math.exp(m)
    except OverflowError:
        return float("inf")


# ---------------------------------------------------------------- คิดทีละตัวอักษร พร้อมบันทึกสูตร
@torch.no_grad()
def think_step(model, tok, ids, temperature=0.8, top_k=20, device="cpu", show=96, force=None):
    """คำนวณตัวอักษรถัดไป 1 ตัว และเก็บตัวเลขทุกขั้นไว้แสดงสูตร
    force = id ของตัวอักษรที่เลือกไว้แล้ว (ใช้ตอนแสดงคำตอบที่ชนะการเลือก) — ตัวเลขทุกขั้นยังคำนวณจริง"""
    model.eval()
    ctx = model.cfg["ctx"]
    ids_c = ids[-ctx:]
    x = torch.tensor([ids_c], device=device)
    logits, _, tr = model(x, trace=True)
    z = logits[0, -1].float().cpu().double()   # GPU ของ Mac ไม่รองรับ float64 จึงย้ายมา CPU ก่อน
    T = max(float(temperature), 1e-3)
    zt = z / T
    expz = torch.exp(zt - zt.max())                 # คำนวณแบบกันตัวเลขล้น
    probs = expz / expz.sum()
    sampling = probs.clone()
    if top_k and top_k < len(probs):
        kth = torch.topk(probs, int(top_k)).values[-1]
        sampling[sampling < kth] = 0
        sampling /= sampling.sum()
    nxt = int(force) if force is not None else int(torch.multinomial(sampling.float(), 1).item())

    top = torch.topk(probs, 8)
    ent = float(-(probs * torch.log2(probs.clamp_min(1e-12))).sum())
    sum_exp = _scaled_sum(zt)   # Σ e^{z/T}

    # attention ชั้นสุดท้าย: แซนด์กำลัง "มอง" ตัวอักษรไหนในอดีต
    last = tr[-1]
    A = last["A"].float().cpu().double()                     # (heads, len)
    avg = A.mean(0)
    n = min(show, len(ids_c))
    pos = int(avg.argmax())
    head = int(A[:, pos].argmax())
    dk = model.cfg["d"] // model.cfg["heads"]
    dot = float(last["dots"][head, pos].cpu())
    row = last["dots"][head].float().cpu().double() / math.sqrt(dk)
    att_sum = _scaled_sum(row)
    return {
        "id": nxt, "ch": tok.itos[nxt],
        "p": float(probs[nxt]), "z": float(z[nxt]), "T": T,
        "sum_exp": sum_exp, "vocab": len(probs),
        "top": [[tok.itos[i], float(probs[i]), float(z[i])] for i in top.indices.tolist()],
        "H": ent,
        "ctx": tok.decode(ids_c[-n:]),
        "att": [round(float(a), 4) for a in avg[-n:]],
        "att_focus": {"ch": tok.itos[ids_c[pos]], "w": float(A[head, pos]), "head": head,
                      "dot": dot, "dk": dk, "sum": att_sum, "back": len(ids_c) - 1 - pos},
        "norms": [round(l["norm"], 2) for l in tr],
    }


# ---------------------------------------------------------------- คิดหลายคำตอบ แล้วเลือกแบบที่เข้ากับบทสนทนาที่สุด
@torch.no_grad()
def sample_replies(model, tok, ids, n=6, temperature=0.8, top_k=20, device="cpu", max_chars=160):
    """สุ่มคำตอบ n แบบพร้อมกัน (หยุดเมื่อเจอขึ้นบรรทัดใหม่) — ยังไม่เลือก"""
    model.eval()
    ctx = model.cfg["ctx"]
    nl = tok.stoi["\n"]
    rows = [list(ids) for _ in range(n)]
    outs = [[] for _ in range(n)]
    alive = list(range(n))
    T = max(float(temperature), 1e-3)
    for _ in range(max_chars):
        if not alive:
            break
        x = torch.tensor([rows[i][-ctx:] for i in alive], device=device)
        z = model(x)[0][:, -1].float() / T
        if top_k and top_k < z.shape[-1]:
            kth = torch.topk(z, int(top_k), dim=-1).values[:, -1:]
            z = z.masked_fill(z < kth, float("-inf"))
        nxt = torch.multinomial(torch.softmax(z, -1), 1)[:, 0].tolist()
        still = []
        for i, c in zip(alive, nxt):
            if c == nl:
                continue
            rows[i].append(c)
            outs[i].append(c)
            still.append(i)
        alive = still
    return [tok.decode(o) for o in outs]


@torch.no_grad()
def reply_logp(model, tok, ctx_ids, reply, device="cpu"):
    """ค่าเฉลี่ย log p ของทุกตัวอักษรในคำตอบ (รวมการจบบรรทัด) เมื่อมีบริบท ctx_ids ก่อนหน้า
    log p(คำตอบ | บริบท) = (1/N) Σᵢ ln p(cᵢ | บริบท, c₁…cᵢ₋₁)"""
    model.eval()
    r = tok.encode(reply + "\n")
    keep = max(1, model.cfg["ctx"] - len(r))
    seq = list(ctx_ids)[-keep:] + r
    seq = seq[-(model.cfg["ctx"] + 1):]
    x = torch.tensor([seq[:-1]], device=device)
    lp = torch.log_softmax(model(x)[0][0].float(), -1)
    n = min(len(r), len(seq) - 1)
    tgt = torch.tensor(seq[-n:], device=device)
    return float(lp[-n:].gather(1, tgt[:, None]).mean())


# ---------------------------------------------------------------- วัดว่าคุยต่อเรื่องได้ไหม
@torch.no_grad()
def topic_test(model, tok, items, device="cpu"):
    """ข้อสอบคุยต่อเรื่อง: แต่ละข้อมีบทสนทนา + คำตอบที่ถูก + คำตอบจากเรื่องอื่น
    ได้คะแนนเมื่อ log p(คำตอบที่ถูก | บทสนทนา) > log p(คำตอบเรื่องอื่น | บทสนทนา)
    คืนค่า % ข้อที่ถูก (สุ่มเดาได้ 50%)"""
    if not items:
        return None
    ok = 0
    for ctx, good, bad in items:
        ids = tok.encode(ctx)
        ok += reply_logp(model, tok, ids, good, device) > reply_logp(model, tok, ids, bad, device)
    return 100.0 * ok / len(items)
