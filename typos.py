"""
คำผิดแบบที่คนพิมพ์จริง — ใช้ 2 ทาง
1) ตอนฝึก: ใส่คำผิดลงฝั่งผู้ใช้ (คำตอบยังถูก) ให้แซนด์เรียนว่าพิมพ์ผิดแบบนี้แปลว่าอะไร
2) ตอนคุย: เดาว่าผู้ใช้ลืมเปลี่ยนแป้นภาษาหรือไม่ (เช่น wfh0ib'q → ได้จริงๆ)
   โดยให้สมองของแซนด์เองตัดสินว่าแบบไหน "เป็นภาษา" มากกว่า:
       เลือกแบบที่ loss ต่ำกว่า   L(x) = −(1/N) Σ ln p(xᵢ | x₍<ᵢ₎)
"""
import random
import re

# แป้นพิมพ์ไทยมาตรฐาน (เกษมณี): ปุ่มอังกฤษ → ตัวไทย
_EN = "1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./" + '!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>?'
_TH = "ๅ/-ภถุึคตจขชๆไำพะัีรนยบลฃฟหกดเ้่าสวงผปแอิืทมใฝ" + "+๑๒๓๔ู฿๕๖๗๘๙๐\"ฎฑธํ๊ณฯญฐ,ฅฤฆฏโฌ็๋ษศซ.()ฉฮฺ์?ฒฬฦ"
assert len(_EN) == len(_TH)
EN2TH = dict(zip(_EN, _TH))
TH2EN = {t: e for e, t in EN2TH.items()}
TH2EN.setdefault(",", "}")

_ROWS = ["1234567890-=", "qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./"]
NEIGHBORS = {}
for r, row in enumerate(_ROWS):
    for c, ch in enumerate(row):
        near = []
        for dr, dc in ((0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < len(_ROWS) and 0 <= cc < len(_ROWS[rr]):
                near.append(_ROWS[rr][cc])
        NEIGHBORS[ch] = near

TH_CONFUSE = ["สศษ", "ทธฑฒ", "คฅฆ", "นณ", "ลฬ", "ขฃ", "พภ", "ตฏ", "ดฎ", "ยญ", "ถฐ", "อฮ", "ซศ"]
TONES = "่้๊๋"
SLANG = [("ครับ", "คับ"), ("ครับ", "ครัช"), ("ค่ะ", "คะ"), ("ค่ะ", "ค่า"), ("อะไร", "อาราย"), ("ไม่", "ม่าย"),
         ("ทำไม", "ทามมาย"), ("เป็น", "เปน"), ("จริงๆ", "จิงๆ"), ("หรือ", "รึ"), ("อย่างไร", "ยังไง"), ("ได้", "ด้าย")]


def is_thai(ch):
    return "฀" <= ch <= "๿"


def to_thai(s):
    return "".join(EN2TH.get(c, c) for c in s)


def to_english(s):
    return "".join(TH2EN.get(c, c) for c in s)


def _neighbor(ch, rnd):
    if is_thai(ch) and ch in TH2EN:
        k = TH2EN[ch].lower()
        if k in NEIGHBORS:
            return EN2TH.get(rnd.choice(NEIGHBORS[k]), ch)
        return ch
    low = ch.lower()
    if low in NEIGHBORS:
        return rnd.choice(NEIGHBORS[low])
    return ch


def noisy(text, rnd=random, n_ops=None):
    """สร้างคำผิดแบบที่คนพิมพ์จริง 1-2 จุด"""
    s = list(text)
    if len(s) < 3:
        return text
    for _ in range(n_ops or rnd.choice((1, 1, 2))):
        op = rnd.random()
        i = rnd.randrange(len(s))
        if op < 0.18:                                   # กดตัวข้างๆ
            s[i] = _neighbor(s[i], rnd)
        elif op < 0.32 and len(s) > 3:                  # ตกตัว
            del s[i]
        elif op < 0.42:                                 # กดซ้ำ
            s.insert(i, s[i])
        elif op < 0.56 and i + 1 < len(s):              # สลับตัว
            s[i], s[i + 1] = s[i + 1], s[i]
        elif op < 0.72:                                 # วรรณยุกต์ผิด/ตก
            idx = [k for k, c in enumerate(s) if c in TONES]
            if idx:
                k = rnd.choice(idx)
                if rnd.random() < 0.5:
                    del s[k]
                else:
                    s[k] = rnd.choice(TONES.replace(s[k], ""))
        elif op < 0.86:                                 # พยัญชนะเสียงเดียวกัน
            idx = [(k, g) for k, c in enumerate(s) for g in TH_CONFUSE if c in g]
            if idx:
                k, g = rnd.choice(idx)
                s[k] = rnd.choice(g.replace(s[k], "") or g)
        else:                                           # ภาษาพูด/สะกดตามเสียง
            t = "".join(s)
            cand = [(a, b) for a, b in SLANG if a in t]
            if cand:
                a, b = rnd.choice(cand)
                s = list(t.replace(a, b, 1))
    return "".join(s)


def wrong_layout(text):
    """จำลองการลืมเปลี่ยนแป้น: พิมพ์ไทยตอนแป้นเป็นอังกฤษ"""
    return to_english(text)


# ---------------------------------------------------------------- เดาตอนคุย
_ASCII_RUN = re.compile(r"[!-~]{3,}")
_LAYOUT_HINT = re.compile(r"[;'\[\],./\\0-9]")


def layout_candidates(msg):
    """ช่วงตัวอักษรอังกฤษที่อาจเป็นภาษาไทยพิมพ์ผิดแป้น"""
    out = []
    for m in _ASCII_RUN.finditer(msg):
        run = m.group(0)
        th = to_thai(run)
        thai_ratio = sum(is_thai(c) for c in th) / len(th)
        if thai_ratio >= 0.6 and run.lower() not in {"http", "https", "www"}:
            out.append((m.start(), m.end(), run, th))
    if not out:                                         # ข้อความไทยที่จริงๆ เป็นภาษาอังกฤษ
        runs = re.finditer(r"[฀-๿]{3,}", msg)
        for m in runs:
            en = to_english(m.group(0))
            if re.fullmatch(r"[A-Za-z ]+", en) and sum(c in "aeiouAEIOU" for c in en) / len(en) > 0.25:
                out.append((m.start(), m.end(), m.group(0), en))
    return out


def english_score(run):
    """ช่วงอังกฤษนี้ดูเป็นคำอังกฤษแค่ไหน (0-1) — ใช้ตอนสมองยังเด็ก"""
    letters = [c for c in run.lower() if c.isalpha()]
    if not letters:
        return 0.0
    vowels = sum(c in "aeiouy" for c in letters) / len(letters)
    weird = len(_LAYOUT_HINT.findall(run)) / len(run)
    return max(0.0, min(1.0, vowels / 0.38)) * (1 - min(1.0, weird * 3))


COMMON_EN = {"internet", "net", "wifi", "hello", "hi", "ok", "okay", "ai", "app", "code", "python", "google", "youtube",
             "facebook", "line", "game", "test", "email", "mail", "online", "web", "website", "file", "computer", "mac",
             "windows", "iphone", "android", "sand", "github", "git", "chat", "bot", "model", "data", "update", "readme"}


def _young_score(seg, kept):
    """สมองยังเด็ก: ให้คะแนนจากรูปแบบตัวอักษร (ส่วนที่แปลงต้องไม่เหมือนอังกฤษ ส่วนที่เหลือต้องเหมือน)"""
    if english_score(seg) >= 0.6:
        return None
    s = (1 - english_score(seg)) * len(seg)
    if kept:
        ks = english_score(kept)
        if kept.lower() not in COMMON_EN and (len(kept) < 4 or ks < 0.6):
            return None
        s += ks * len(kept) + (6 if kept.lower() in COMMON_EN else 0)
    return s


def guess_layout(msg, lm_loss, young=False, margin=0.4):
    """ลองแปลงแป้นทีละช่วง (ลองทุกจุดตัด เพราะคำอังกฤษจริงอาจติดอยู่ เช่น internetwfh0ib'q)
    แล้วให้สมองแซนด์ตัดสิน: แบบที่ loss ต่ำกว่าชัดเจนคือคำตอบ
    คืน (ข้อความที่แก้แล้ว, รายการการแก้)"""
    fixes = []
    fixed = msg
    for m in reversed(list(_ASCII_RUN.finditer(msg))):
        run, start, end = m.group(0), m.start(), m.end()
        if run.lower() in COMMON_EN or run.lower().startswith(("http", "www")):
            continue
        cuts = [(0, len(run))] + [(k, len(run)) for k in range(2, len(run) - 2)] + \
               [(0, k) for k in range(3, len(run) - 1)]
        base = None if young else lm_loss(fixed)
        best = None
        for a, b in cuts:
            seg = run[a:b]
            conv = to_thai(seg)
            if len(seg) < 3 or sum(is_thai(c) for c in conv) / len(conv) < 0.6:
                continue
            kept = run[:a] + run[b:]
            cand_run = run[:a] + conv + run[b:]
            cand = fixed[:start] + cand_run + fixed[end:]
            sc = _young_score(seg, kept)          # ด่านแรก: ต้องดูเหมือนพิมพ์ผิดแป้นจริง
            if sc is None:
                continue
            if young:
                if best is None or sc > best[0]:
                    best = (sc, cand, seg, conv, None)
            else:
                loss = lm_loss(cand)
                rank = None if loss is None else -loss + 0.05 * sc   # สมองตัดสินเป็นหลัก รูปแบบตัวอักษรช่วยนิดหน่อย
                if rank is not None and (best is None or rank > best[0]):
                    best = (rank, cand, seg, conv, loss)
        if best is None:
            continue
        if young or (base is not None and best[4] < base - margin):
            fixes.append({"from": best[2], "to": best[3], "loss_before": base, "loss_after": best[4]})
            fixed = best[1]
    return fixed, fixes[::-1]
