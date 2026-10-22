#!/usr/bin/env python3
"""
แซนด์ (Sand) — AI ที่สร้างขึ้นเองตั้งแต่ศูนย์ รันและเรียนรู้บนเครื่องคุณ 100%
  brain.py    สมอง Transformer ที่เขียนเองทุกสูตร
  trainer.py  ครูฝึกที่ฝึกสมองอยู่เบื้องหลัง
  formulas.py เครื่องคำนวณสูตรการเติบโต (แก้สูตรได้จากในแอป)
รัน: python app.py แล้วเปิด http://127.0.0.1:7777
"""
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import webbrowser
from html import escape as html_escape
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import bigbrain as B
import formulas as F
import seed_data
from brain import SAND_TAG, USER_TAG, think_step
from trainer import Trainer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "sand.db")
BRAIN_PATH = os.path.join(BASE, "sand_brain.pt")
PORT = int(os.environ.get("SAND_PORT", "7777"))
MAX_REPLY = 200
MODELS_DIR = os.path.join(BASE, "models")
SCHEMA = "2"

# ---------------------------------------------------------------- สูตร
# (key, ชื่อ, สูตร, หน่วย, คำอธิบาย, kind)  kind: state = แสดงการเติบโต, train = ครูฝึกใช้จริงตอนฝึก
BUILTIN_FORMULAS = [
    ("xp", "ค่าประสบการณ์ (XP)", "steps / 20 + chats * 5 + taught * 20 + likes * 10", "XP",
     "ได้ XP จากการฝึก การคุย สิ่งที่สอน และคำชม", "state"),
    ("level", "เลเวล", "floor(sqrt(xp / 40)) + 1", "",
     "ใช้รากที่สอง ยิ่งเลเวลสูงยิ่งต้องใช้ XP มากขึ้นเรื่อยๆ", "state"),
    ("skill", "ความเก่งในการเดาตัวอักษร", "clamp(100 * (1 - val_loss / log(vocab)), 0, 100)", "%",
     "0% = เดาสุ่ม, 100% = เดาถูกแน่นอนทุกตัว วัดจากข้อความที่แซนด์ไม่ได้ใช้ฝึก "
     "(ยิ่งข้อมูลฝึกง่ายและซ้ำ ค่านี้ยิ่งสูง ไม่ได้แปลว่าฉลาดเท่าคน)", "state"),
    ("ppl", "ความลังเล (perplexity)", "exp(val_loss)", "ตัวเลือก",
     "โดยเฉลี่ยแซนด์ลังเลระหว่างตัวอักษรกี่ตัว ถ้าเดาสุ่มจะเท่ากับจำนวนตัวอักษรทั้งหมด", "state"),
    ("grains", "เม็ดทรายในลูกแก้ว", "30 + level * 15 + skill * 3", "เม็ด",
     "จำนวนเม็ดทรายในลูกแก้วมุมซ้ายบน (แสดงได้สูงสุด 480)", "state"),
    ("lr", "อัตราการเรียนรู้ η", "0.001 * min(1, steps / 300) * (0.1 + 0.9 * exp(-steps / 30000))", "",
     "ขนาดก้าวตอนปรับน้ำหนักสมอง ช่วงแรกค่อยๆ เร่ง (warmup) แล้วค่อยๆ ลดเพื่อให้เรียนละเอียดขึ้น "
     "ครูฝึกใช้สูตรนี้ทุกก้าว", "train"),
    ("w_taught", "น้ำหนัก: สิ่งที่สอนโดยตรง", "8", "เท่า",
     "สิ่งที่สอนและคำตอบที่แก้ให้ จะถูกอ่านซ้ำกี่เท่าต่อรอบข้อมูล", "train"),
    ("w_liked", "น้ำหนัก: คำตอบที่ได้ 👍", "4", "เท่า", "คำตอบที่ได้ 👍 จะถูกอ่านซ้ำกี่เท่า", "train"),
    ("w_text", "น้ำหนัก: ข้อความที่ป้อน", "1", "เท่า", "ข้อความยาวที่ป้อนให้อ่าน จะถูกอ่านซ้ำกี่เท่า", "train"),
]

VAR_LABELS = {
    "steps": "จำนวนก้าวที่ฝึกแล้ว",
    "tokens_m": "จำนวนตัวอักษรที่อ่านตอนฝึก (ล้าน)",
    "loss": "loss ตอนฝึก (ค่าเฉลี่ยเคลื่อนที่)",
    "val_loss": "loss บนข้อความที่ไม่ได้ใช้ฝึก",
    "best_loss": "val_loss ที่ดีที่สุดเท่าที่เคยได้",
    "vocab": "จำนวนตัวอักษรที่แซนด์รู้จัก",
    "params_m": "จำนวนน้ำหนักในสมอง (ล้าน)",
    "chats": "จำนวนข้อความที่ผู้ใช้ส่ง",
    "taught": "จำนวนคู่ถาม-ตอบที่สอน",
    "texts_k": "ข้อความที่ป้อนให้อ่าน (พันตัวอักษร)",
    "likes": "จำนวน 👍",
    "dislikes": "จำนวน 👎",
    "days": "อายุของแซนด์ (วัน)",
}

lock = threading.Lock()
trainer = None
SEED_TEXT = seed_data.build()


# ---------------------------------------------------------------- database
def db():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def q(sql, args=()):
    c = db()
    try:
        return [dict(r) for r in c.execute(sql, args).fetchall()]
    finally:
        c.close()


def x(sql, args=()):
    with lock:
        c = db()
        try:
            cur = c.execute(sql, args)
            c.commit()
            return cur.lastrowid
        finally:
            c.close()


def count(sql, args=()):
    return q(sql, args)[0]["n"] or 0


def setting(key, default=None):
    r = q("SELECT value FROM settings WHERE key=?", (key,))
    return r[0]["value"] if r else default


def set_setting(key, value):
    x("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, str(value)))


def event(kind, title):
    x("INSERT INTO events(kind,title,ts) VALUES(?,?,?)", (kind, title, time.time()))


def init_db():
    c = db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, role TEXT, content TEXT, ts REAL, rating INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, kind TEXT, title TEXT, ts REAL);
    CREATE TABLE IF NOT EXISTS growth(id INTEGER PRIMARY KEY, ts REAL, xp INTEGER, memories INTEGER, messages INTEGER, traits TEXT);
    CREATE TABLE IF NOT EXISTS taught(id INTEGER PRIMARY KEY, q TEXT, a TEXT, ts REAL, source TEXT DEFAULT 'teach');
    CREATE TABLE IF NOT EXISTS texts(id INTEGER PRIMARY KEY, title TEXT, content TEXT, ts REAL);
    CREATE TABLE IF NOT EXISTS traces(msg_id INTEGER PRIMARY KEY, data TEXT);
    """)
    old = c.execute("SELECT value FROM settings WHERE key='schema'").fetchone()
    if not old or old[0] != SCHEMA:
        # อัปเกรดจากเวอร์ชันที่ใช้ Ollama: สูตรเดิมใช้ตัวแปรคนละชุด จึงเริ่มสูตรใหม่
        c.execute("DROP TABLE IF EXISTS formulas")
        c.execute("INSERT OR REPLACE INTO settings VALUES('schema',?)", (SCHEMA,))
    c.execute("""CREATE TABLE IF NOT EXISTS formulas(id INTEGER PRIMARY KEY, key TEXT UNIQUE, name TEXT, expr TEXT,
        default_expr TEXT, description TEXT, unit TEXT DEFAULT '', builtin INTEGER DEFAULT 0, kind TEXT DEFAULT 'state', ts REAL)""")
    if "formulas" not in [r[1] for r in c.execute("PRAGMA table_info(growth)")]:
        c.execute("ALTER TABLE growth ADD COLUMN formulas TEXT")
    for key, name, expr, unit, desc, kind in BUILTIN_FORMULAS:
        c.execute("INSERT OR IGNORE INTO formulas(key,name,expr,default_expr,description,unit,builtin,kind,ts) "
                  "VALUES(?,?,?,?,?,?,1,?,?)", (key, name, expr, expr, desc, unit, kind, time.time()))
    if not c.execute("SELECT 1 FROM events LIMIT 1").fetchone():
        c.execute("INSERT INTO events(kind,title,ts) VALUES('birth','แซนด์ถือกำเนิดขึ้น',?)", (time.time(),))
    c.commit()
    c.close()


# ---------------------------------------------------------------- ตัวแปรและสูตร
def base_vars():
    t = trainer
    rl = math.log(len(t.tok))
    return {
        "steps": t.step,
        "tokens_m": t.tokens / 1e6,
        "loss": t.ema if t.ema is not None else rl,
        "val_loss": t.val_loss if t.val_loss is not None else rl,
        "best_loss": t.best_val if t.best_val is not None else rl,
        "vocab": len(t.tok),
        "params_m": t.model.n_params() / 1e6,
        "chats": count("SELECT COUNT(*) n FROM messages WHERE role='user'"),
        "taught": count("SELECT COUNT(*) n FROM taught"),
        "texts_k": (q("SELECT SUM(LENGTH(content)) n FROM texts")[0]["n"] or 0) / 1000,
        "likes": count("SELECT COUNT(*) n FROM messages WHERE rating>0"),
        "dislikes": count("SELECT COUNT(*) n FROM messages WHERE rating<0"),
        "days": (time.time() - t.born) / 86400,
    }


def formula_rows():
    return q("SELECT * FROM formulas ORDER BY id")


def eval_row(f, v):
    for e in (f["expr"], f["default_expr"]):
        if e:
            try:
                return F.evaluate(F.parse(e), v)
            except F.FormulaError:
                continue
    return 0.0


def core():
    v = base_vars()
    res = {}
    for f in formula_rows():
        v[f["key"]] = res[f["key"]] = eval_row(f, v)
    lv = max(1, int(math.floor(res.get("level", 1))))
    return v, res, lv


def xp_threshold(level, v):
    row = q("SELECT * FROM formulas WHERE key='level'")
    if not row:
        return None
    try:
        node = F.parse(row[0]["expr"])
    except F.FormulaError:
        node = F.parse(row[0]["default_expr"])

    def lv_at(xp):
        try:
            return math.floor(F.evaluate(node, {**v, "xp": xp}))
        except F.FormulaError:
            return -1
    if lv_at(0) >= level:
        return 0
    hi = 1.0
    while lv_at(hi) < level:
        hi *= 2
        if hi > 1e12:
            return None
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if lv_at(mid) >= level:
            hi = mid
        else:
            lo = mid
    return math.ceil(hi)


def var_labels():
    labels = dict(VAR_LABELS)
    for f in formula_rows():
        labels[f["key"]] = f"ผลของสูตร: {f['name']}"
    return labels


def allowed_names(fid):
    return set(VAR_LABELS) | {r["key"] for r in formula_rows() if fid is None or r["id"] < fid}


def check_formula(expr, fid=None):
    node = F.parse(expr)
    allowed = allowed_names(fid)
    unknown = F.names(node) - allowed
    if unknown:
        raise F.FormulaError("ไม่รู้จักตัวแปร: " + ", ".join(sorted(unknown)) +
                             " (ใช้ได้เฉพาะตัวแปรพื้นฐานและสูตรที่อยู่ก่อนหน้า)")
    v = core()[0]
    vv = {k: v[k] for k in allowed if k in v}
    return node, F.evaluate(node, vv), vv


def preview_formula(d):
    fid = int(d["id"]) if d.get("id") else None
    try:
        node, val, vv = check_formula(d.get("expr", ""), fid)
    except F.FormulaError as e:
        return {"ok": False, "error": str(e)}
    used = F.names(node)
    return {"ok": True, "value": val, "html": F.render(node, labels=var_labels()),
            "html_sub": F.render(node, vals={k: vv[k] for k in used if k in vv})}


KEY_RE = re.compile(r"^[a-z_][a-z0-9_]{0,23}$")


def save_formula(d):
    expr = (d.get("expr") or "").strip()
    if d.get("id"):
        row = q("SELECT * FROM formulas WHERE id=?", (int(d["id"]),))
        if not row:
            return "ไม่พบสูตร"
        f = row[0]
        try:
            check_formula(expr, f["id"])
        except F.FormulaError as e:
            return str(e)
        if f["builtin"]:
            x("UPDATE formulas SET expr=? WHERE id=?", (F.normalize(expr), f["id"]))
        else:
            name = (d.get("name") or "").strip()[:60]
            if not name:
                return "ต้องตั้งชื่อสูตร"
            x("UPDATE formulas SET name=?,expr=?,description=?,unit=? WHERE id=?",
              (name, F.normalize(expr), (d.get("description") or "").strip()[:200], (d.get("unit") or "").strip()[:12], f["id"]))
        event("formula", f"แก้สูตร: {f['name']}")
        after_formula_change()
        return None
    key = (d.get("key") or "").strip().lower()
    name = (d.get("name") or "").strip()[:60]
    if not name:
        return "ต้องตั้งชื่อสูตร"
    if not KEY_RE.match(key):
        return "ชื่อตัวแปรต้องเป็นภาษาอังกฤษตัวเล็ก ตัวเลข หรือ _ (ขึ้นต้นด้วยตัวอักษร)"
    reserved = set(VAR_LABELS) | set(F.FUNCS) | set(F.CONSTS) | {"if", "_if", "and", "or", "not"}
    if key in reserved or q("SELECT 1 FROM formulas WHERE key=?", (key,)):
        return f"ชื่อตัวแปร '{key}' ถูกใช้แล้ว"
    try:
        check_formula(expr, None)
    except F.FormulaError as e:
        return str(e)
    x("INSERT INTO formulas(key,name,expr,default_expr,description,unit,builtin,kind,ts) VALUES(?,?,?,?,?,?,0,'state',?)",
      (key, name, F.normalize(expr), "", (d.get("description") or "").strip()[:200], (d.get("unit") or "").strip()[:12], time.time()))
    event("formula", f"สร้างสูตรใหม่: {name}")
    record_growth()
    return None


def after_formula_change():
    LR["t"] = 0           # ให้ครูฝึกโหลดสูตร lr ใหม่ทันที
    trainer.dirty = True  # น้ำหนักข้อมูลอาจเปลี่ยน
    record_growth()


def delete_formula(fid):
    row = q("SELECT * FROM formulas WHERE id=? AND builtin=0", (fid,))
    if not row:
        return "ลบสูตรในตัวไม่ได้"
    key = row[0]["key"]
    for f in formula_rows():
        if f["id"] != fid:
            try:
                if key in F.names(F.parse(f["expr"])):
                    return f"สูตร \"{f['name']}\" ใช้ตัวแปร {key} อยู่ ต้องแก้สูตรนั้นก่อน"
            except F.FormulaError:
                pass
    x("DELETE FROM formulas WHERE id=?", (fid,))
    event("formula", f"ลบสูตร: {row[0]['name']}")
    return None


def plot_formula(d):
    fid = int(d["id"]) if d.get("id") else None
    try:
        node, cur_y, vv = check_formula(d.get("expr", ""), fid)
        lo, hi = float(d.get("lo", 0)), float(d.get("hi", 10))
    except (F.FormulaError, TypeError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    var = d.get("var")
    if var not in vv or hi <= lo:
        return {"ok": False, "error": "เลือกตัวแปรและช่วงให้ถูกต้อง"}
    pts = []
    for i in range(121):
        xv = lo + (hi - lo) * i / 120
        try:
            pts.append([xv, F.evaluate(node, {**vv, var: xv})])
        except F.FormulaError:
            pts.append([xv, None])
    return {"ok": True, "points": pts, "current": [vv[var], cur_y], "label": var_labels().get(var, var)}


def formulas_state(v, res):
    rows = q("SELECT formulas FROM growth WHERE formulas IS NOT NULL ORDER BY id DESC LIMIT 600")[::-1]
    step = max(1, len(rows) // 60)
    hist = []
    for r in rows[::step]:
        try:
            hist.append(json.loads(r["formulas"]))
        except Exception:
            pass
    labels = var_labels()
    out = []
    for f in formula_rows():
        item = dict(f)
        try:
            node = F.parse(f["expr"])
            val = F.evaluate(node, v)
            used = sorted(F.names(node))
            item.update(error=None, value=val, html=F.render(node, labels=labels),
                        html_sub=F.render(node, vals={k: v[k] for k in used if k in v}),
                        used=[{"name": k, "label": labels.get(k, k), "value": v.get(k)} for k in used])
        except F.FormulaError as e:
            item.update(error=f"{e} — ตอนนี้ใช้สูตรเดิมแทน" if f["default_expr"] else str(e),
                        value=res.get(f["key"]), html=html_escape(f["expr"]), html_sub="", used=[])
        item["history"] = [h[f["key"]] for h in hist if f["key"] in h]
        out.append(item)
    variables = [{"name": k, "label": VAR_LABELS[k], "value": v[k]} for k in VAR_LABELS]
    return {"list": out, "variables": variables, "funcs": F.FUNC_HELP}


# ---------------------------------------------------------------- ข้อมูลฝึก + อัตราการเรียนรู้
LR = {"t": 0, "node": None, "v": {}}


def get_lr(step):
    """ครูฝึกเรียกทุกก้าว: ใช้สูตร lr ที่ผู้ใช้แก้ได้"""
    if time.time() - LR["t"] > 3:
        LR["t"] = time.time()
        try:
            LR["v"] = core()[0]
            row = q("SELECT * FROM formulas WHERE key='lr'")[0]
            try:
                LR["node"] = F.parse(row["expr"])
                F.evaluate(LR["node"], {**LR["v"], "steps": step})
            except F.FormulaError:
                LR["node"] = F.parse(row["default_expr"])
        except Exception as e:
            print("[lr]", e)
    try:
        lr = F.evaluate(LR["node"], {**LR["v"], "steps": step})
    except Exception:
        lr = 1e-3
    return max(0.0, min(0.05, lr))


def pair(qq, a):
    return f"{USER_TAG}{qq}\n{SAND_TAG}{a}\n"


def liked_pairs():
    return q("""SELECT a.id, a.content ans, (SELECT u.content FROM messages u WHERE u.id < a.id AND u.role='user'
                ORDER BY u.id DESC LIMIT 1) qn FROM messages a WHERE a.role='assistant' AND a.rating>0""")


def get_corpus():
    res = core()[1]
    w = lambda k, dflt: max(0.0, min(50.0, res.get(k, dflt)))
    taught = [pair(r["q"], r["a"]) for r in q("SELECT q,a FROM taught")]
    texts = [r["content"] for r in q("SELECT content FROM texts")]
    liked = [pair(r["qn"], r["ans"]) for r in liked_pairs() if r["qn"]]
    # ข้อความที่ไม่ได้ใช้ฝึก (validation) ตัดจากท้ายสุด จึงวางบทเรียนตั้งต้นไว้ท้าย
    unique = "\n".join(texts + liked + taught + [SEED_TEXT])
    weighted = [(t, w("w_taught", 8) - 1) for t in taught] + \
               [(t, w("w_liked", 4) - 1) for t in liked] + \
               [(t, w("w_text", 1) - 1) for t in texts]
    return weighted, unique


def data_summary():
    res = core()[1]
    seed = len(SEED_TEXT)
    t_chars = q("SELECT SUM(LENGTH(q)+LENGTH(a)) n FROM taught")[0]["n"] or 0
    x_chars = q("SELECT SUM(LENGTH(content)) n FROM texts")[0]["n"] or 0
    l_chars = sum(len(r["ans"]) + len(r["qn"] or "") for r in liked_pairs())
    return {
        "sources": [
            {"name": "บทเรียนตั้งต้น", "chars": seed, "weight": 1},
            {"name": "สิ่งที่สอน", "chars": t_chars, "weight": res.get("w_taught")},
            {"name": "คำตอบที่ได้ 👍", "chars": l_chars, "weight": res.get("w_liked")},
            {"name": "ข้อความที่ป้อน", "chars": x_chars, "weight": res.get("w_text")},
        ],
        "taught": q("SELECT id,q,a,ts,source FROM taught ORDER BY id DESC LIMIT 300"),
        "texts": q("SELECT id,title,LENGTH(content) chars,substr(content,1,120) preview,ts FROM texts ORDER BY id DESC"),
        "seed_preview": SEED_TEXT[:600],
    }


def data_changed():
    trainer.dirty = True
    record_growth()


# ---------------------------------------------------------------- การเติบโต
def check_milestones():
    v, res, lv = core()
    last = int(setting("last_level", "1"))
    if lv > last:
        set_setting("last_level", lv)
        event("level", f"เลเวลอัป! แซนด์ขึ้นเป็นเลเวล {lv}")
    for kind, val, marks, fmt in (
        ("skill", res.get("skill", 0), (10, 25, 50, 70, 80, 90, 95), "เดาตัวอักษรเก่งขึ้นถึง {}% แล้ว"),
        ("steps", v["steps"], (100, 1000, 5000, 10000, 50000, 100000, 500000), "ฝึกครบ {:,} ก้าว"),
        ("chat", v["chats"], (1, 10, 50, 100, 500), "คุยกันครบ {} ข้อความ"),
        ("teach", v["taught"], (1, 10, 50, 100), "ได้รับการสอนครบ {} เรื่อง"),
    ):
        done = float(setting("ms_" + kind, "0"))
        for m in marks:
            if val >= m > done:
                event(kind, fmt.format(m))
                done = m
        set_setting("ms_" + kind, done)
    return v, res


def record_growth():
    v, res = check_milestones()
    x("INSERT INTO growth(ts,xp,memories,messages,traits,formulas) VALUES(?,?,?,?,?,?)",
      (time.time(), round(res.get("xp", 0)), v["taught"], v["chats"], "{}", json.dumps(res)))


# ---------------------------------------------------------------- คุย: คิดทีละตัวอักษรพร้อมส่งสูตรแบบสด
def gen_settings():
    return {
        "temperature": float(setting("temperature", "0.7")),
        "top_k": int(float(setting("top_k", "20"))),
        "delay": int(float(setting("delay", "40"))),
        "auto_train": setting("auto_train", "1") == "1",
        "think": setting("think", "1") == "1",
        "max_tokens": int(float(setting("max_tokens", "1024"))),
        "brain": setting("brain", "baby"),
    }


def build_prompt(ctx):
    hist = q("SELECT role,content FROM messages ORDER BY id DESC LIMIT 12")[::-1]
    text = ""
    for m in hist:
        text += (USER_TAG if m["role"] == "user" else SAND_TAG) + m["content"].replace("\n", " ") + "\n"
    text += SAND_TAG
    return text[-(ctx - 1):]


# ---------------------------------------------------------------- สมองโต (โมเดลเปิด + เครื่องยนต์ของเราเอง)
class BigBrain:
    def __init__(self):
        self.model = self.tok = self.key = None
        self.lock = threading.Lock()
        self.dl = {k: {"state": "idle", "done": 0, "total": 0, "file": "", "error": None} for k in B.MODELS}
        self.stops = {}
        self.load_state = {"state": "idle", "progress": 0.0, "error": None, "key": None}

    def ready(self):
        return self.model is not None

    def status(self):
        ram = B.system_ram_gb()
        out = []
        for k, m in B.MODELS.items():
            out.append({**m, "key": k, "downloaded": B.is_downloaded(MODELS_DIR, k), "dl": self.dl[k],
                        "fits": ram is None or ram + 0.5 >= m["ram_gb"],
                        "quant8": k == "typhoon-4b" and ram is not None and ram + 0.5 < m["ram_gb"]})
        info = None
        if self.model:
            mm = self.model
            info = {"key": self.key, "label": B.MODELS[self.key]["label"], "params": mm.n_params, "layers": mm.L,
                    "heads": mm.H, "kv": mm.KV, "d": mm.d, "hd": mm.hd, "device": mm.device,
                    "dtype": str(mm.dtype).replace("torch.", ""), "quant8": mm.quant8}
        return {"models": out, "loaded": info, "load": self.load_state, "ram_gb": ram}

    def download(self, key):
        if key not in B.MODELS or self.dl[key]["state"] == "downloading":
            return
        stop = threading.Event()
        self.stops[key] = stop

        def job():
            try:
                if B.download(MODELS_DIR, key, self.dl[key], stop):
                    event("brain", f"ดาวน์โหลดสมอง {B.MODELS[key]['label']} เสร็จแล้ว")
            except Exception as e:
                self.dl[key].update(state="error", error=f"{type(e).__name__}: {e} — ตรวจการเชื่อมต่ออินเทอร์เน็ตแล้วกดดาวน์โหลดต่อ")
        threading.Thread(target=job, daemon=True).start()

    def stop_download(self, key):
        if key in self.stops:
            self.stops[key].set()

    def load(self, key):
        if key not in B.MODELS or not B.is_downloaded(MODELS_DIR, key) or self.load_state["state"] == "loading":
            return
        self.load_state = {"state": "loading", "progress": 0.0, "error": None, "key": key}

        def job():
            try:
                if trainer.running:
                    trainer.stop()              # หยุดฝึกแซนด์ทารก เพื่อให้แรมพอสำหรับสมองโต
                    set_setting("auto_train", "0")
                self.unload()
                ram = B.system_ram_gb()
                q8 = key == "typhoon-4b" and ram is not None and ram + 0.5 < B.MODELS[key]["ram_gb"]
                model = B.Qwen3(B.model_dir(MODELS_DIR, key), quant8=q8,
                                progress=lambda f: self.load_state.update(progress=f))
                tok = B.TokenizerWrap(B.model_dir(MODELS_DIR, key))
                with self.lock:
                    self.model, self.tok, self.key = model, tok, key
                set_setting("big_model", key)
                set_setting("brain", "big")
                self.load_state.update(state="ready", progress=1.0)
                event("brain", f"แซนด์ได้สมองโต: {B.MODELS[key]['label']} ({model.n_params / 1e9:.1f} พันล้านน้ำหนัก)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.load_state.update(state="error", error=f"{type(e).__name__}: {e}")
        threading.Thread(target=job, daemon=True).start()

    def unload(self):
        with self.lock:
            self.model = self.tok = self.key = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


BIG = BigBrain()

SYS_BASE = (
    "คุณคือ \"แซนด์\" (Sand) ผู้ช่วย AI ที่ทำงานบนคอมพิวเตอร์ของผู้ใช้แบบออฟไลน์ทั้งหมด "
    "เครื่องยนต์คิดของคุณเขียนขึ้นเองโดยผู้ใช้ "
    "ตอบเป็นภาษาเดียวกับที่ผู้ใช้ถาม (ส่วนใหญ่เป็นภาษาไทย) ใช้คำลงท้ายว่า ครับ "
    "ตอบให้ถูกต้อง ชัดเจน และเป็นประโยชน์ อธิบายเหตุผลเป็นขั้นตอนเมื่อเป็นเรื่องซับซ้อน "
    "ถ้าไม่แน่ใจให้บอกตรงๆ ห้ามแต่งข้อมูลขึ้นเอง"
)

_CHUNKS = {"key": None, "chunks": []}


def _grams(s):
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i + 3] for i in range(len(s) - 2)}


def retrieve(question, k=2):
    """ค้นข้อความที่ผู้ใช้ป้อนไว้ ส่วนที่เกี่ยวกับคำถามที่สุด (เทียบตัวอักษร 3 ตัวที่ซ้ำกัน)"""
    key = q("SELECT COUNT(*) n, MAX(id) m FROM texts")[0]
    key = (key["n"], key["m"])
    if _CHUNKS["key"] != key:
        chunks = []
        for t in q("SELECT title,content FROM texts"):
            c = t["content"]
            for i in range(0, max(1, len(c)), 450):
                piece = c[i:i + 600]
                if len(piece.strip()) > 40:
                    chunks.append((t["title"], piece, _grams(piece)))
        _CHUNKS.update(key=key, chunks=chunks)
    qg = _grams(question)
    if len(qg) < 3:
        return []
    scored = sorted(((len(qg & g) / math.sqrt(len(g) + 1), title, piece) for title, piece, g in _CHUNKS["chunks"]),
                    reverse=True)[:k]
    return [(title, piece) for sc, title, piece in scored if sc > 0.35]


def build_big_prompt(user_text, think_on, tk, budget=3000):
    sys_txt = SYS_BASE
    taught = q("SELECT q,a FROM taught ORDER BY id DESC LIMIT 60")[::-1]
    if taught:
        sys_txt += "\n\nสิ่งที่ผู้ใช้สอนคุณไว้ (เชื่อถือได้ ใช้ตอบได้เลย):\n" + "\n".join(f"- ถาม: {t['q']} → ตอบ: {t['a']}" for t in taught)
    found = retrieve(user_text)
    if found:
        sys_txt += "\n\nข้อความที่ผู้ใช้เคยให้อ่าน ส่วนที่อาจเกี่ยวข้อง:\n" + "\n---\n".join(f"[{t}] {p}" for t, p in found)
    hist = q("SELECT role,content FROM messages ORDER BY id DESC LIMIT 9")[::-1]
    head = f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
    turns = [f"<|im_start|>{'user' if m['role'] == 'user' else 'assistant'}\n{m['content']}<|im_end|>\n" for m in hist]
    tail = "<|im_start|>assistant\n" + ("" if think_on else "<think>\n\n</think>\n\n")
    while len(turns) > 1 and len(tk.encode(head + "".join(turns) + tail)) > budget:
        turns.pop(0)
    return head + "".join(turns) + tail, [t for t, _ in found], len(taught)


def _stream_big(user_text, write):
    gs = gen_settings()
    x("INSERT INTO messages(role,content,ts) VALUES('user',?,?)", (user_text, time.time()))
    with BIG.lock:
        m, tk, key = BIG.model, BIG.tok, BIG.key
    meta = B.MODELS[key]
    think_on = meta["think"] and gs["think"]
    prompt, found, n_taught = build_big_prompt(user_text, think_on, tk)
    ids = tk.encode(prompt)
    T, top_k, top_p, rp = gs["temperature"], gs["top_k"], 0.95, 1.05
    info = {"mode": "big", "size_label": meta["label"], "label": meta["label"], "params": m.n_params,
            "device": m.device, "layers": m.L, "heads": m.H, "kv": m.KV, "d": m.d, "hd": m.hd,
            "step": trainer.step, "T": T, "top_k": top_k, "top_p": top_p, "rep": rp, "think": think_on,
            "prompt_tokens": len(ids), "prompt_chars": len(ids), "taught": n_taught, "retrieved": found,
            "quant8": m.quant8, "vocab": tk.vocab}
    write({"start": info, "prompt": prompt[-4000:]})
    stops = {i for i in (tk.id("<|im_end|>"), tk.id("<|endoftext|>")) if i is not None}
    t_s, t_e = tk.id("<think>"), tk.id("</think>")
    cache = m.new_cache()
    t0 = time.time()
    with BIG.lock:
        logits, tr = m.forward(ids, 0, cache, trace=True)
    seq, gen, tokens = list(ids), [], []
    phase = "answer"
    buf = {"think": [], "answer": []}
    shown = {"think": "", "answer": ""}
    for _ in range(max(16, gs["max_tokens"])):
        nxt, st = B.sample_step(logits, tr, tk, T, top_k, top_p, rp, gen, ctx_ids=seq)
        gen.append(nxt)
        st["phase"] = phase
        if nxt in stops:
            st["stop"] = True
            tokens.append(st)
            write({"tok": st})
            break
        if nxt == t_s:
            phase = st["phase"] = "think"
            st["marker"], st["ch"] = "think_start", ""
        elif nxt == t_e:
            st["marker"], st["ch"] = "think_end", ""
            phase = "answer"
        else:
            buf[phase].append(nxt)
            full = tk.decode(buf[phase])
            if full.endswith("\ufffd"):
                st["ch"] = ""                       # รอไบต์ที่เหลือของตัวอักษรนี้
            else:
                st["ch"] = full[len(shown[phase]):]
                shown[phase] = full
        tokens.append(st)
        write({"tok": st})
        seq.append(nxt)
        with BIG.lock:
            logits, tr = m.forward([nxt], len(seq) - 1, cache, trace=True)
        if gs["delay"]:
            time.sleep(gs["delay"] / 1000)
    secs = time.time() - t0
    answer = shown["answer"].strip() or "…"
    ps = [t["p"] for t in tokens]
    summary = {"chars": len(answer), "tokens": len(tokens), "secs": round(secs, 2),
               "avg_p": sum(ps) / len(ps) if ps else 0,
               "ppl": math.exp(-sum(math.log(max(p, 1e-12)) for p in ps) / len(ps)) if ps else 0,
               "avg_H": sum(t["H"] for t in tokens) / len(tokens) if tokens else 0,
               "think_tokens": len(buf["think"])}
    aid = x("INSERT INTO messages(role,content,ts) VALUES('assistant',?,?)", (answer, time.time()))
    x("INSERT OR REPLACE INTO traces(msg_id,data) VALUES(?,?)",
      (aid, json.dumps({"info": info, "prompt": prompt[-4000:], "tokens": tokens[:1500],
                        "thinking": shown["think"], "summary": summary}, ensure_ascii=False)))
    write({"done": True, "id": aid, "summary": summary})
    record_growth()


def stream_chat(user_text, write):
    if gen_settings()["brain"] == "big":
        if not BIG.ready():
            st = BIG.load_state["state"]
            write({"error": "สมองโตกำลังโหลดอยู่ รอสักครู่แล้วถามใหม่" if st == "loading"
                   else "ยังไม่ได้โหลดสมองโต — กดปุ่มสมองที่มุมขวาบนเพื่อเลือก หรือสลับกลับไปใช้แซนด์ทารก"})
            return
        try:
            _stream_big(user_text, write)
        except Exception as e:
            import traceback
            traceback.print_exc()
            write({"error": f"{type(e).__name__}: {e}"})
        return
    trainer.waiting += 1
    try:
        _stream_chat(user_text, write)
    except Exception as e:
        import traceback
        traceback.print_exc()
        write({"error": f"{type(e).__name__}: {e}"})
    finally:
        trainer.waiting -= 1


def _stream_chat(user_text, write):
    gs = gen_settings()
    x("INSERT INTO messages(role,content,ts) VALUES('user',?,?)", (user_text.replace("\n", " "), time.time()))
    t = trainer
    ctx = t.model.cfg["ctx"]
    prompt = build_prompt(ctx)
    ids = t.tok.encode(prompt)
    info = {"size": t.size, "size_label": t.snapshot()["size_label"], "params": t.model.n_params(),
            "device": t.device, "ctx": ctx, "layers": t.model.cfg["layers"], "heads": t.model.cfg["heads"],
            "d": t.model.cfg["d"], "step": t.step, "T": gs["temperature"], "top_k": gs["top_k"],
            "prompt_chars": len(prompt), "vocab": len(t.tok)}
    write({"start": info, "prompt": prompt})
    tokens, out, t0 = [], "", time.time()
    for _ in range(MAX_REPLY):
        with t.lock:
            st = think_step(t.model, t.tok, ids, gs["temperature"], gs["top_k"], t.device)
        if st["ch"] == "\n":
            st["stop"] = True
            tokens.append(st)
            write({"tok": st})
            break
        ids.append(st["id"])
        out += st["ch"]
        tokens.append(st)
        write({"tok": st})
        if gs["delay"]:
            time.sleep(gs["delay"] / 1000)
    secs = time.time() - t0
    ps = [tk["p"] for tk in tokens]
    summary = {"chars": len(out), "secs": round(secs, 2),
               "avg_p": sum(ps) / len(ps) if ps else 0,
               "ppl": math.exp(-sum(math.log(max(p, 1e-12)) for p in ps) / len(ps)) if ps else 0,
               "avg_H": sum(tk["H"] for tk in tokens) / len(tokens) if tokens else 0}
    aid = x("INSERT INTO messages(role,content,ts) VALUES('assistant',?,?)", (out or "…", time.time()))
    x("INSERT OR REPLACE INTO traces(msg_id,data) VALUES(?,?)",
      (aid, json.dumps({"info": info, "prompt": prompt, "tokens": tokens, "summary": summary}, ensure_ascii=False)))
    write({"done": True, "id": aid, "summary": summary})
    record_growth()


def rate(mid, val):
    row = q("SELECT rating FROM messages WHERE id=? AND role='assistant'", (mid,))
    if not row:
        return 0
    val = 0 if val == row[0]["rating"] else max(-1, min(1, val))
    x("UPDATE messages SET rating=? WHERE id=?", (val, mid))
    data_changed()
    return val


def correct(mid, text):
    text = (text or "").strip().replace("\n", " ")[:400]
    qn = q("SELECT content FROM messages WHERE id<? AND role='user' ORDER BY id DESC LIMIT 1", (mid,))
    if not text or not qn:
        return "ไม่มีคำตอบหรือไม่พบคำถาม"
    x("INSERT INTO taught(q,a,ts,source) VALUES(?,?,?,'correct')", (qn[0]["content"], text, time.time()))
    event("teach", "ผู้ใช้สอนคำตอบที่ถูกต้อง")
    data_changed()
    return None


def state():
    v, res, lv = core()
    xp = res.get("xp", 0)
    born = q("SELECT ts FROM events ORDER BY id LIMIT 1")
    return {
        "level": lv, "xp": round(xp, 1), "xp_cur": xp_threshold(lv, v) or 0, "xp_next": xp_threshold(lv + 1, v),
        "grains": res.get("grains", 60), "skill": res.get("skill", 0),
        "formulas": formulas_state(v, res),
        "train": trainer.snapshot(),
        "settings": gen_settings(),
        "data": data_summary(),
        "events": q("SELECT kind,title,ts FROM events ORDER BY id DESC LIMIT 150"),
        "born": born[0]["ts"] if born else time.time(),
        "chats": v["chats"],
        "big": BIG.status(),
    }


# ---------------------------------------------------------------- web server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(BASE, "static", "index.html"), "rb") as f:
                b = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/api/state":
            self.send_json(state())
        elif u.path == "/api/history":
            self.send_json(q("SELECT m.id,m.role,m.content,m.ts,m.rating,(t.msg_id IS NOT NULL) has_trace FROM messages m "
                             "LEFT JOIN traces t ON t.msg_id=m.id ORDER BY m.id DESC LIMIT 60")[::-1])
        elif u.path == "/api/trace":
            mid = int(parse_qs(u.query).get("id", ["0"])[0])
            r = q("SELECT data FROM traces WHERE msg_id=?", (mid,))
            self.send_json(json.loads(r[0]["data"]) if r else {"error": "ไม่มีข้อมูล"})
        else:
            self.send_error(404)

    def do_POST(self):
        d = self.body()
        p = self.path
        ok = lambda err=None: self.send_json({"ok": not err, "error": err}, 400 if err else 200)
        if p == "/api/chat":
            text = (d.get("message") or "").strip()
            if not text:
                return ok("ข้อความว่าง")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            alive = [True]

            def write(o):
                if not alive[0]:
                    return
                try:
                    self.wfile.write((json.dumps(o, ensure_ascii=False) + "\n").encode())
                    self.wfile.flush()
                except Exception:
                    alive[0] = False
            stream_chat(text, write)
        elif p == "/api/rate":
            self.send_json({"rating": rate(int(d.get("id", 0)), int(d.get("value", 0)))})
        elif p == "/api/correct":
            ok(correct(int(d.get("id", 0)), d.get("text")))
        elif p == "/api/teach":
            qq, a = (d.get("q") or "").strip().replace("\n", " ")[:400], (d.get("a") or "").strip().replace("\n", " ")[:400]
            if not qq or not a:
                return ok("ต้องมีทั้งคำถามและคำตอบ")
            x("INSERT INTO taught(q,a,ts) VALUES(?,?,?)", (qq, a, time.time()))
            event("teach", "ผู้ใช้สอนเรื่องใหม่ให้แซนด์")
            data_changed()
            ok()
        elif p == "/api/teach/delete":
            x("DELETE FROM taught WHERE id=?", (int(d.get("id", 0)),))
            data_changed()
            ok()
        elif p == "/api/text/add":
            content = (d.get("content") or "").strip()
            if len(content) < 20:
                return ok("ข้อความสั้นเกินไป (อย่างน้อย 20 ตัวอักษร)")
            if len(content) > 5_000_000:
                return ok("ข้อความยาวเกินไป (สูงสุด 5 ล้านตัวอักษร)")
            title = (d.get("title") or content[:30]).strip()[:80]
            x("INSERT INTO texts(title,content,ts) VALUES(?,?,?)", (title, content, time.time()))
            event("text", f"ป้อนข้อความให้อ่าน: {title} ({len(content):,} ตัวอักษร)")
            data_changed()
            ok()
        elif p == "/api/text/delete":
            x("DELETE FROM texts WHERE id=?", (int(d.get("id", 0)),))
            data_changed()
            ok()
        elif p == "/api/train/start":
            trainer.start()
            set_setting("auto_train", "1")
            ok()
        elif p == "/api/train/stop":
            threading.Thread(target=trainer.stop, daemon=True).start()
            set_setting("auto_train", "0")
            ok()
        elif p == "/api/train/reset":
            size = d.get("size", "medium")
            was = trainer.running
            trainer.reset(size)
            event("birth", f"แซนด์เกิดใหม่ด้วยสมองขนาด{trainer.snapshot()['size_label']} "
                           f"({trainer.model.n_params():,} น้ำหนัก)")
            for k in ("last_level", "ms_skill", "ms_steps"):
                set_setting(k, "0" if k != "last_level" else "1")
            record_growth()
            if was:
                trainer.start()
            ok()
        elif p == "/api/brain":
            mode = d.get("mode")
            if mode in ("baby", "big"):
                set_setting("brain", mode)
            ok()
        elif p == "/api/big/download":
            BIG.download(d.get("model"))
            ok()
        elif p == "/api/big/stop":
            BIG.stop_download(d.get("model"))
            ok()
        elif p == "/api/big/load":
            BIG.load(d.get("model"))
            ok()
        elif p == "/api/big/unload":
            BIG.unload()
            set_setting("brain", "baby")
            BIG.load_state = {"state": "idle", "progress": 0.0, "error": None, "key": None}
            ok()
        elif p == "/api/settings":
            if "think" in d:
                set_setting("think", "1" if d["think"] else "0")
            for k, lo, hi in (("temperature", 0.05, 3), ("top_k", 1, 200), ("delay", 0, 1000), ("max_tokens", 16, 4096)):
                if k in d:
                    try:
                        set_setting(k, max(lo, min(hi, float(d[k]))))
                    except (TypeError, ValueError):
                        pass
            ok()
        elif p == "/api/formulas/preview":
            self.send_json(preview_formula(d))
        elif p == "/api/formulas/save":
            ok(save_formula(d))
        elif p == "/api/formulas/delete":
            ok(delete_formula(int(d.get("id", 0))))
        elif p == "/api/formulas/reset":
            x("UPDATE formulas SET expr=default_expr WHERE id=? AND builtin=1", (int(d.get("id", 0)),))
            after_formula_change()
            ok()
        elif p == "/api/formulas/plot":
            self.send_json(plot_formula(d))
        else:
            self.send_error(404)


def main():
    global trainer
    init_db()
    trainer = Trainer(BRAIN_PATH, get_corpus, get_lr, record_growth)
    s = trainer.snapshot()
    url = f"http://127.0.0.1:{PORT}"
    dev = {"mps": "GPU ของ Mac (Metal)", "cuda": "การ์ดจอ NVIDIA", "cpu": "CPU"}[trainer.device]
    print("=" * 56)
    print("  แซนด์ (Sand) — AI ที่สร้างเองตั้งแต่ศูนย์")
    print(f"  เปิดเบราว์เซอร์ที่: {url}")
    print(f"  สมองขนาด{s['size_label']} {s['params']:,} น้ำหนัก · ฝึกด้วย {dev} · ฝึกไปแล้ว {s['step']:,} ก้าว")
    print("  กด Ctrl+C เพื่อปิด (สมองจะถูกบันทึกอัตโนมัติ)")
    print("=" * 56)
    gs = gen_settings()
    bm = setting("big_model")
    if gs["brain"] == "big" and bm in B.MODELS and B.is_downloaded(MODELS_DIR, bm):
        print(f"  กำลังโหลดสมองโต {B.MODELS[bm]['label']} …")
        BIG.load(bm)
    elif gs["auto_train"]:
        trainer.start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    if not os.environ.get("SAND_NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nกำลังบันทึกสมอง…")
        trainer.stop()
        print("แซนด์ไปพักแล้ว สมองถูกบันทึกไว้ใน sand_brain.pt")


if __name__ == "__main__":
    main()
