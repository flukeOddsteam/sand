"""
สมองโตของแซนด์ — เครื่องยนต์ที่เขียนเองทั้งหมด สำหรับโมเดลสถาปัตยกรรม Qwen3
ใช้น้ำหนักที่ฝึกมาแล้วจากโมเดลเปิด (open weights) แต่ทุกสูตรในการคิดเขียนเองที่นี่ ไม่ใช้ Ollama / transformers

ทุกชั้น (layer):
  h = RMSNorm(x)            = x / √(mean(x²) + ε) · γ
  q, k, v = h·W_q, h·W_k, h·W_v          (k, v ใช้หัวร่วมกัน: Grouped-Query Attention)
  q, k = RMSNorm(q), RMSNorm(k)           (ต่อหัว)
  q, k = RoPE(q), RoPE(k)                 (หมุนเวกเตอร์ตามตำแหน่ง)
  A = softmax(q·kᵀ / √d_k + mask)
  x = x + (A·v)·W_o
  h = RMSNorm(x)
  x = x + (SiLU(h·W_gate) ⊙ (h·W_up))·W_down    (SwiGLU)
logits = RMSNorm(x)·W_outᵀ
"""
import json
import math
import mmap
import os
import struct
import threading
import time
import urllib.error
import urllib.request
import warnings

import torch
import torch.nn.functional as Fn

HF = "https://huggingface.co"

MODELS = {
    "qwen3-0.6b": {"repo": "Qwen/Qwen3-0.6B", "label": "Qwen3 0.6B", "params": "0.6 พันล้าน",
                   "size_gb": 1.5, "ram_gb": 8, "think": True,
                   "desc": "เร็วและเบาเครื่อง ตอบเรื่องพื้นฐานได้ คิดก่อนตอบได้"},
    "qwen3-1.7b": {"repo": "Qwen/Qwen3-1.7B", "label": "Qwen3 1.7B", "params": "1.7 พันล้าน",
                   "size_gb": 4.1, "ram_gb": 8, "think": True,
                   "desc": "แนะนำสำหรับ RAM 8GB · ฉลาดกว่า 0.6B มาก · คิดก่อนตอบได้ (เห็นความคิดสด)"},
    "typhoon-4b": {"repo": "typhoon-ai/typhoon2.5-qwen3-4b", "label": "Typhoon 2.5 (Qwen3 4B)", "params": "4 พันล้าน",
                   "size_gb": 8.1, "ram_gb": 16, "think": False,
                   "desc": "ปรับจูนภาษาไทยโดยเฉพาะ เก่งที่สุดในชุดนี้ · ต้อง RAM 16GB (ถ้าน้อยกว่าจะบีบเป็น int8 ช้าและเครื่องตึง)"},
}
CONFIG_FILES = ["config.json", "tokenizer.json"]
OPTIONAL_FILES = ["tokenizer_config.json", "generation_config.json"]


def system_ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device):
    try:
        a = torch.ones(4, 4, device=device, dtype=torch.bfloat16)
        (a @ a).sum().item()
        return torch.bfloat16
    except Exception:
        return torch.float16 if device != "cpu" else torch.float32


# ---------------------------------------------------------------- ดาวน์โหลดน้ำหนัก (ต่อได้ถ้าหลุด)
def _open(url, start=0):
    headers = {"User-Agent": "sand-ai/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def _size(url):
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "sand-ai/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def model_dir(root, key):
    return os.path.join(root, key)


def weight_files(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    return ["model.safetensors"]


def is_downloaded(root, key):
    d = model_dir(root, key)
    if not all(os.path.exists(os.path.join(d, f)) for f in CONFIG_FILES):
        return False
    return all(os.path.exists(os.path.join(d, f)) for f in weight_files(d))


def download(root, key, status, stop):
    """ดาวน์โหลดไฟล์โมเดลจาก Hugging Face ลงโฟลเดอร์ models/<key> (ต่อจากเดิมได้)"""
    repo = MODELS[key]["repo"]
    d = model_dir(root, key)
    os.makedirs(d, exist_ok=True)
    url = lambda f: f"{HF}/{repo}/resolve/main/{f}"
    status.update(state="downloading", done=0, total=0, file="รายชื่อไฟล์", error=None)
    files = list(CONFIG_FILES)
    for f in OPTIONAL_FILES:
        files.append((f, True))
    try:
        with _open(url("model.safetensors.index.json")) as r:
            data = r.read()
        with open(os.path.join(d, "model.safetensors.index.json"), "wb") as f:
            f.write(data)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    files += weight_files(d)
    plan = [(f, False) if isinstance(f, str) else f for f in files]
    sizes = {f: _size(url(f)) for f, _ in plan}
    status["total"] = sum(sizes.values())
    done = 0
    for f, optional in plan:
        dest = os.path.join(d, f)
        if os.path.exists(dest) and (not sizes[f] or os.path.getsize(dest) == sizes[f]):
            done += os.path.getsize(dest)
            status["done"] = done
            continue
        part = dest + ".part"
        start = os.path.getsize(part) if os.path.exists(part) else 0
        status["file"] = f
        try:
            r = _open(url(f), start)
        except urllib.error.HTTPError as e:
            if optional and e.code == 404:
                continue
            raise
        if start and r.status != 206:     # เซิร์ฟเวอร์ไม่รองรับการต่อ เริ่มใหม่
            start = 0
        done += start
        with r, open(part, "ab" if start else "wb") as out:
            while True:
                if stop.is_set():
                    status.update(state="idle", file="หยุดไว้ (กดดาวน์โหลดต่อได้)")
                    return False
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                status["done"] = done
        os.replace(part, dest)
    status.update(state="done", file="", done=status["total"])
    return True


# ---------------------------------------------------------------- อ่านไฟล์ safetensors เอง
_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def read_safetensors(path):
    """รูปแบบไฟล์: [ขนาด header 8 ไบต์][header JSON][ข้อมูลดิบของทุก tensor]"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    base = 8 + n
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = _DT[info["dtype"]]
        s, e = info["data_offsets"]
        esize = torch.tensor([], dtype=dt).element_size()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = torch.frombuffer(mm, dtype=dt, count=(e - s) // esize, offset=base + s)
        yield name, t.reshape(info["shape"])


# ---------------------------------------------------------------- สมอง Qwen3
class Qwen3:
    def __init__(self, path, device=None, dtype=None, quant8=False, progress=None):
        self.path = path
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        self.c = c
        self.device = device or pick_device()
        self.dtype = dtype or pick_dtype(self.device)
        self.d = c["hidden_size"]
        self.H = c["num_attention_heads"]
        self.KV = c["num_key_value_heads"]
        self.hd = c.get("head_dim") or self.d // self.H
        self.L = c["num_hidden_layers"]
        self.eps = c.get("rms_norm_eps", 1e-6)
        self.theta = c.get("rope_theta", 1e6)
        self.tied = c.get("tie_word_embeddings", False)
        self.quant8 = quant8
        self.W = {}
        self.n_params = 0
        files = weight_files(path)
        for i, f in enumerate(files):
            for name, t in read_safetensors(os.path.join(path, f)):
                name = name[6:] if name.startswith("model.") else name
                self.n_params += t.numel()
                if quant8 and t.dim() == 2 and "embed_tokens" not in name and name != "lm_head.weight":
                    tf = t.float()
                    scale = tf.abs().amax(dim=1).clamp_min(1e-8) / 127.0          # ต่อแถว (ต่อช่องผลลัพธ์)
                    q = torch.round(tf / scale[:, None]).clamp(-127, 127).to(torch.int8)
                    self.W[name] = (q.to(self.device), scale.to(self.device, self.dtype))
                else:
                    self.W[name] = t.to(self.device, self.dtype).clone() if self.device == "cpu" else t.to(self.device, self.dtype)
            if progress:
                progress((i + 1) / len(files))
        if "lm_head.weight" not in self.W:
            self.W["lm_head.weight"] = self.W["embed_tokens.weight"]
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        self.inv_freq = inv.to(self.device)

    # ----- ชิ้นส่วนสูตร
    def lin(self, x, name):
        w = self.W[name]
        if isinstance(w, tuple):                     # int8: y = (x·W8ᵀ) ⊙ scale
            q, s = w
            return (x @ q.to(self.dtype).T) * s
        return x @ w.T

    def rms(self, x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return w * xf.to(self.dtype)

    def rope(self, x, cos, sin):
        h = x.shape[-1] // 2
        rot = torch.cat((-x[..., h:], x[..., :h]), dim=-1)
        return x * cos + rot * sin

    def new_cache(self):
        return [[None, None] for _ in range(self.L)]

    @torch.no_grad()
    def forward(self, ids, pos0, cache, trace=False):
        """ids: โทเคนใหม่ (list), pos0: ตำแหน่งเริ่ม, cache: K/V ของโทเคนก่อนหน้า → logits ของตำแหน่งสุดท้าย"""
        T = len(ids)
        x = self.W["embed_tokens.weight"][torch.tensor(ids, device=self.device)]
        pos = torch.arange(pos0, pos0 + T, device=self.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(self.dtype)[:, None, :], emb.sin().to(self.dtype)[:, None, :]
        scale = 1.0 / math.sqrt(self.hd)
        rep = self.H // self.KV
        tr = [] if trace else None
        for i in range(self.L):
            p = f"layers.{i}."
            h = self.rms(x, self.W[p + "input_layernorm.weight"])
            q = self.lin(h, p + "self_attn.q_proj.weight").view(T, self.H, self.hd)
            k = self.lin(h, p + "self_attn.k_proj.weight").view(T, self.KV, self.hd)
            v = self.lin(h, p + "self_attn.v_proj.weight").view(T, self.KV, self.hd)
            if p + "self_attn.q_norm.weight" in self.W:
                q = self.rms(q, self.W[p + "self_attn.q_norm.weight"])
                k = self.rms(k, self.W[p + "self_attn.k_norm.weight"])
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            if cache[i][0] is not None:
                k = torch.cat((cache[i][0], k), 0)
                v = torch.cat((cache[i][1], v), 0)
            cache[i][0], cache[i][1] = k, v
            S = k.shape[0]
            K = k.repeat_interleave(rep, dim=1).transpose(0, 1)        # (H, S, hd)
            V = v.repeat_interleave(rep, dim=1).transpose(0, 1)
            Q = q.transpose(0, 1)                                       # (H, T, hd)
            if T > 1:
                o = Fn.scaled_dot_product_attention(Q[None], K[None], V[None], is_causal=(pos0 == 0),
                                                    attn_mask=None if pos0 == 0 else self._mask(T, S, pos0))[0]
                if trace:
                    dots = (Q[:, -1:] @ K.transpose(1, 2))[:, 0].float()
                    A = torch.softmax(dots * scale, dim=-1)
            else:
                dots = (Q @ K.transpose(1, 2)).float()                  # q·k
                A = torch.softmax(dots * scale, dim=-1)                 # softmax(q·k/√d_k)
                o = A.to(self.dtype) @ V
                dots, A = dots[:, 0], A[:, 0]
            x = x + self.lin(o.transpose(0, 1).reshape(T, self.H * self.hd), p + "self_attn.o_proj.weight")
            h = self.rms(x, self.W[p + "post_attention_layernorm.weight"])
            g = self.lin(h, p + "mlp.gate_proj.weight")
            u = self.lin(h, p + "mlp.up_proj.weight")
            x = x + self.lin(Fn.silu(g) * u, p + "mlp.down_proj.weight")
            if trace:
                tr.append({"A": A, "dots": dots, "norm": float(x[-1].float().norm()), "hd": self.hd})
        x = self.rms(x[-1:], self.W["norm.weight"])
        logits = self.lin(x, "lm_head.weight")[0]
        return logits, tr

    def _mask(self, T, S, pos0):
        i = torch.arange(T, device=self.device)[:, None] + pos0
        j = torch.arange(S, device=self.device)[None, :]
        return (j <= i)[None]


# ---------------------------------------------------------------- ตัวตัดคำ + คิดทีละโทเคน
class TokenizerWrap:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.vocab = self.tk.get_vocab_size(with_added_tokens=True)
        self.id = lambda s: self.tk.token_to_id(s)

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=False)

    def piece(self, i):
        return self.tk.decode([i], skip_special_tokens=False)


def _safe_sum(v):
    try:
        return float(torch.exp(v - v.max()).sum()) * math.exp(float(v.max()))
    except OverflowError:
        return float("inf")


def sample_step(logits, tr, tok, T, top_k, top_p, rep_pen, history, show=48, ctx_ids=None):
    """จากคะแนนดิบ z → ความน่าจะเป็น → สุ่มโทเคน พร้อมเก็บตัวเลขทุกขั้นไว้แสดงสูตร"""
    z = logits.float().cpu().double()
    if rep_pen and rep_pen != 1.0 and history:
        idx = torch.tensor(sorted(set(history)), dtype=torch.long)
        zz = z[idx]
        z[idx] = torch.where(zz > 0, zz / rep_pen, zz * rep_pen)
    T = max(float(T), 1e-3)
    zt = z / T
    m = float(zt.max())
    ez = torch.exp(zt - m)
    probs = ez / ez.sum()
    samp = probs.clone()
    if top_k and top_k < len(samp):
        kth = torch.topk(samp, int(top_k)).values[-1]
        samp[samp < kth] = 0
    if top_p and top_p < 1.0:
        sp, si = torch.sort(samp, descending=True)
        cum = torch.cumsum(sp / sp.sum(), 0)
        cut = si[(cum - sp / sp.sum()) > top_p]
        samp[cut] = 0
    samp /= samp.sum()
    nxt = int(torch.multinomial(samp.float(), 1).item())
    top = torch.topk(probs, 8)
    ent = float(-(probs * torch.log2(probs.clamp_min(1e-12))).sum())
    try:
        sum_exp = float(ez.sum()) * math.exp(m)
    except OverflowError:
        sum_exp = float("inf")
    last = tr[-1]
    A = last["A"].float().cpu().double()               # (H, S)
    avg = A.mean(0)
    S = len(avg)
    n = min(show, S)
    sub = avg[S - n:]
    j = int(sub[1:].argmax()) + 1 if n > 1 else 0     # ข้ามโทเคนแรกสุด (มักเป็นจุดพักของ attention)
    pos = S - n + j
    head = int(A[:, pos].argmax())
    dots = last["dots"].float().cpu().double()
    row = dots[head] / math.sqrt(last.get("hd", 128))
    ctx_ids = ctx_ids or []
    ctx_tokens = [tok.piece(i) for i in ctx_ids[-n:]]
    return nxt, {
        "id": nxt, "ch": tok.piece(nxt), "p": float(probs[nxt]), "z": float(z[nxt]), "T": T,
        "sum_exp": sum_exp, "vocab": len(probs),
        "top": [[tok.piece(i), float(probs[i]), float(z[i])] for i in top.indices.tolist()],
        "H": ent,
        "ctx": ctx_tokens, "att": [round(float(a), 4) for a in sub],
        "att_focus": {"ch": ctx_tokens[j] if ctx_tokens else "", "w": float(A[head, pos]), "head": head,
                      "dot": float(dots[head, pos]), "dk": last.get("hd", 128),
                      "sum": _safe_sum(row),
                      "back": S - 1 - pos},
        "norms": [round(l["norm"], 1) for l in tr],
    }


@torch.no_grad()
def quick_text(model, tok, prompt, max_new=20, stop_text=("\n",)):
    """เขียนข้อความสั้นๆ แบบเลือกตัวที่น่าจะเป็นที่สุดทุกครั้ง (ใช้ให้สมองโตคิดคำค้น)"""
    ids = tok.encode(prompt)
    stops = {i for i in (tok.id("<|im_end|>"), tok.id("<|endoftext|>")) if i is not None}
    cache = model.new_cache()
    logits, _ = model.forward(ids, 0, cache)
    out = []
    for _ in range(max_new):
        nxt = int(logits.float().argmax())
        if nxt in stops:
            break
        out.append(nxt)
        text = tok.decode(out)
        if any(s in text for s in stop_text):
            break
        logits, _ = model.forward([nxt], len(ids) + len(out) - 1, cache)
    text = tok.decode(out)
    for s in stop_text:
        text = text.split(s)[0]
    return text.strip()
