"""
ครูฝึกของแซนด์ — ฝึกสมองอยู่เบื้องหลังตลอดเวลาที่กดเริ่ม
ทุกก้าว (step):
  1) สุ่มข้อความยาว ctx ตัวอักษรมา batch ชิ้น
  2) ให้แซนด์เดาตัวอักษรถัดไปทุกตำแหน่ง
  3) loss  L = −(1/N) Σ ln p(ตัวที่ถูก)
  4) คำนวณ gradient ∂L/∂θ แล้วปรับน้ำหนักด้วย AdamW:  θ ← θ − η · m̂ / (√v̂ + ε)
"""
import math
import os
import random
import threading
import time

import torch

from brain import SIZES, SAND_TAG, Brain, Tokenizer, USER_TAG, pick_device, think_step, topic_test

SAMPLE_PROMPTS = ["สวัสดี", "คุณชื่ออะไร", "3 + 4 เท่ากับเท่าไร", "คำตรงข้ามของ ร้อน", "cat แปลว่าอะไร",
                  "วันต่อจากวันจันทร์คือวันอะไร", "ขอบคุณ", "นับ 2 ถึง 6"]


class Trainer:
    MAX_VOCAB = 4000

    def __init__(self, path, get_corpus, get_lr, on_progress, on_vocab=None):
        self.path = path
        self.get_corpus = get_corpus      # () -> [(text, weight), ...] , unique_text
        self.get_lr = get_lr              # (step) -> learning rate (จากสูตร lr)
        self.on_progress = on_progress    # เรียกทุก 100 ก้าว
        self.on_vocab = on_vocab          # เรียกเมื่อสมองรู้จักตัวอักษรใหม่
        self.device = pick_device()
        self.tok = Tokenizer()
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        self.waiting = 0          # มีคนรอคุยอยู่กี่คน (ครูฝึกจะหลีกทางให้)
        self.dirty = True
        self.train_data = self.val_data = self.chat_data = None
        self.p_chat = 0.0
        self.topic_items = []     # ข้อสอบคุยต่อเรื่อง (ตั้งจาก app.py)
        self.topic = None
        self.topic_hist = []
        self.data_info = {}
        self.ema = None
        self.last = {}
        self.speed = 0.0
        self.error = None
        if not self.load():
            self.new("medium")

    # ------------------------------------------------------------ สร้าง/โหลด/บันทึก
    def new(self, size):
        cfg = SIZES.get(size, SIZES["medium"])
        self.size = size if size in SIZES else "medium"
        self.model = Brain(len(self.tok), **cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
        self.batch = cfg["batch"]
        self.step = 0
        self.tokens = 0
        self.history = []        # [step, train_loss, val_loss, lr, gnorm, ts]
        self.samples = []        # [step, prompt, text]
        self.val_loss = None
        self.best_val = None
        self.born = time.time()
        self.ema = None
        self.topic = None
        self.topic_hist = []

    def load(self):
        if not os.path.exists(self.path):
            return False
        try:
            ck = torch.load(self.path, map_location="cpu", weights_only=False)
            self.tok = Tokenizer(ck["chars"])
            self.size = ck["size"]
            self.model = Brain(**ck["cfg"])
            self.model.load_state_dict(ck["model"])
            self.model.to(self.device)
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
            try:
                self.opt.load_state_dict(ck["opt"])
            except Exception:
                pass
            self.batch = SIZES[self.size]["batch"]
            self.step, self.tokens = ck["step"], ck["tokens"]
            self.history, self.samples = ck["history"], ck["samples"]
            self.val_loss, self.best_val = ck.get("val_loss"), ck.get("best_val")
            self.born = ck.get("born", time.time())
            self.topic_hist = ck.get("topic_hist", [])
            self.topic = self.topic_hist[-1][1] if self.topic_hist else None
            self.ema = self.history[-1][1] if self.history else None
            return True
        except Exception as e:
            print("[โหลดสมองไม่สำเร็จ เริ่มใหม่]", e)
            return False

    def save(self):
        with self.lock:
            ck = {"cfg": self.model.cfg, "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
                  "opt": self.opt.state_dict(), "chars": self.tok.itos, "size": self.size,
                  "step": self.step, "tokens": self.tokens, "history": self.history[-4000:],
                  "samples": self.samples[-300:], "val_loss": self.val_loss, "best_val": self.best_val,
                  "born": self.born, "topic_hist": self.topic_hist[-400:]}
        tmp = self.path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, self.path)

    def reset(self, size):
        self.stop()
        with self.lock:
            self.new(size)
            self.dirty = True
        self.save()

    # ------------------------------------------------------------ ข้อมูล
    def grow_vocab(self, text, min_count=3):
        """เจอตัวอักษรที่ไม่รู้จักบ่อยพอ → ขยายสมองให้รู้จัก (ภาษาใหม่)"""
        from collections import Counter
        cnt = Counter(c for c in text if c not in self.tok.stoi and c.isprintable() and not c.isspace())
        room = self.MAX_VOCAB - len(self.tok)
        new = [c for c, n in cnt.most_common() if n >= min_count][:max(0, room)]
        if not new:
            return []
        self.tok.add_chars(new)
        self.model.expand_vocab(len(new))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
        if self.on_vocab:
            try:
                self.on_vocab(new)
            except Exception as e:
                print("[on_vocab]", e)
        return new

    def build_data(self):
        got = self.get_corpus()
        weighted, unique = got[0], got[1]
        extra_info = got[2] if len(got) > 2 else {}
        self.grow_vocab(unique + "".join(t for t, w in weighted[:2000]))
        ids = self.tok.encode(unique)
        cut = int(len(ids) * 0.95)
        base, val = ids[:cut], ids[cut:]
        extra = []
        for text, w in weighted:
            reps = int(w) + (1 if random.random() < w - int(w) else 0)
            extra += self.tok.encode(text) * max(0, reps)
        need = self.model.cfg["ctx"] + 2
        train = base + extra
        if len(train) < need:
            train = (train or self.tok.encode(" ")) * (need // max(1, len(train)) + 2)
        if len(val) < need:
            val = train[-need * 4:]
        self.train_data = torch.tensor(train, dtype=torch.long)
        self.val_data = torch.tensor(val, dtype=torch.long)
        # บทสนทนาแยกไว้อีกกอง: ทุกก้าวหยิบจากกองนี้ตามสัดส่วน p_chat ไม่ให้ถูกบทความยาวๆ กลบ
        chat = self.tok.encode(extra_info.get("chat", ""))
        self.chat_data = torch.tensor(chat, dtype=torch.long) if len(chat) > need * 2 else None
        self.p_chat = max(0.0, min(1.0, float(extra_info.get("p_chat", 0.0))))
        self.data_info = {"train_chars": len(train), "val_chars": len(val), "unique_chars": len(unique),
                          "chat_chars": len(chat), "p_chat": self.p_chat}
        self.dirty = False

    def get_batch(self, data, n):
        ctx = self.model.cfg["ctx"]
        ix = torch.randint(0, len(data) - ctx - 1, (n,))
        x = torch.stack([data[i:i + ctx] for i in ix])
        y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    # ------------------------------------------------------------ ฝึก
    def train_step(self):
        lr = self.get_lr(self.step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train()
        n_chat = int(round(self.batch * self.p_chat)) if self.chat_data is not None else 0
        parts = []
        if n_chat:
            parts.append(self.get_batch(self.chat_data, n_chat))
        if self.batch - n_chat:
            parts.append(self.get_batch(self.train_data, self.batch - n_chat))
        x = torch.cat([a for a, _ in parts])
        y = torch.cat([b for _, b in parts])
        _, loss, _ = self.model(x, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
        self.opt.step()
        self.step += 1
        self.tokens += x.numel()
        lv = float(loss.detach())
        self.ema = lv if self.ema is None else 0.97 * self.ema + 0.03 * lv
        self.last = {"loss": lv, "lr": lr, "gnorm": gnorm}

    @torch.no_grad()
    def evaluate(self, n=6):
        self.model.eval()
        tot = 0.0
        for _ in range(n):
            x, y = self.get_batch(self.val_data, min(self.batch, 16))
            tot += float(self.model(x, y)[1].detach())
        return tot / n

    def speak(self, prompt, max_chars=80, temperature=0.7):
        ids = self.tok.encode(f"{USER_TAG}{prompt}\n{SAND_TAG}")
        out = ""
        for _ in range(max_chars):
            st = think_step(self.model, self.tok, ids, temperature, 20, self.device)
            if st["ch"] == "\n":
                break
            ids.append(st["id"])
            out += st["ch"]
        return out

    def loop(self):
        t0, s0 = time.time(), self.step
        try:
            while self.running:
                if self.waiting:
                    time.sleep(0.02)   # หยุดฝึกชั่วคราวระหว่างแซนด์กำลังตอบ
                    continue
                with self.lock:
                    if self.dirty:
                        self.build_data()
                    self.train_step()
                    if self.step % 100 == 0:
                        self.val_loss = self.evaluate()
                        self.best_val = self.val_loss if self.best_val is None else min(self.best_val, self.val_loss)
                        self.history.append([self.step, round(self.ema, 4), round(self.val_loss, 4),
                                             self.last["lr"], round(self.last["gnorm"], 3), time.time()])
                        if self.step % 500 == 0 and self.topic_items:
                            self.topic = topic_test(self.model, self.tok, self.topic_items, self.device)
                            self.topic_hist.append([self.step, round(self.topic, 1)])
                        if self.step % 300 == 0:
                            p = SAMPLE_PROMPTS[(self.step // 300) % len(SAMPLE_PROMPTS)]
                            self.samples.append([self.step, p, self.speak(p)])
                if self.step % 100 == 0:
                    self.on_progress()
                if self.step % 1000 == 0:
                    self.save()
                now = time.time()
                if now - t0 > 2:
                    self.speed = (self.step - s0) / (now - t0)
                    t0, s0 = now, self.step
                time.sleep(0.0005)   # เปิดช่องให้การคุยแทรกเข้ามาได้
        except Exception as e:
            self.error = str(e)
            self.running = False
            print("[ฝึกผิดพลาด]", e)
        self.speed = 0.0
        self.save()

    def start(self):
        if self.running:
            return
        self.error = None
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=30)
            self.thread = None

    def snapshot(self):
        return {
            "running": self.running, "device": self.device, "size": self.size,
            "size_label": SIZES[self.size]["label"], "cfg": self.model.cfg,
            "params": self.model.n_params(), "vocab": len(self.tok),
            "step": self.step, "tokens": self.tokens, "batch": self.batch,
            "loss": self.ema, "val_loss": self.val_loss, "best_val": self.best_val,
            "last": self.last, "speed": round(self.speed, 2), "error": self.error,
            "random_loss": math.log(len(self.tok)), "born": self.born,
            "history": self.history[-400:], "samples": self.samples[-40:][::-1],
            "data": self.data_info, "dirty": self.dirty,
            "topic": self.topic, "topic_hist": self.topic_hist[-100:],
            "sizes": {k: {"label": v["label"], "d": v["d"], "layers": v["layers"], "heads": v["heads"], "ctx": v["ctx"]}
                      for k, v in SIZES.items()},
        }
