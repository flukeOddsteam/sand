"""
ประตูสู่อินเทอร์เน็ตของแซนด์ — อ่านจาก Wikipedia ผ่าน API ทางการเท่านั้น
- ไม่เกิน 1 คำขอต่อวินาที และแนะนำตัวด้วย User-Agent ตามนโยบายของ Wikimedia
- เนื้อหา Wikipedia ใช้สัญญาอนุญาต CC BY-SA: แซนด์เก็บชื่อบทความและลิงก์ที่มาไว้ทุกครั้ง
"""
import html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request

API = os.environ.get("SAND_WIKI_API", "https://{lang}.wikipedia.org/w/api.php")
UA = "SandAI/1.0 (personal local AI learning project; https://github.com/flukeOddsteam/sand)"
_lock = threading.Lock()
_last = [0.0]


def _api(lang, **params):
    params.update(format="json", formatversion="2", utf8="1")
    url = API.format(lang=lang) + "?" + urllib.parse.urlencode(params)
    with _lock:
        wait = 1.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def clean(text):
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        m = re.match(r"^=+\s*(.*?)\s*=+$", line)
        if m:
            line = m.group(1)
            if line in ("ดูเพิ่ม", "อ้างอิง", "แหล่งข้อมูลอื่น", "เชิงอรรถ", "See also", "References", "External links", "Notes"):
                break                                   # ส่วนท้ายบทความไม่ค่อยมีเนื้อหา
        lines.append(line)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or ""))


def search(q, lang="th", n=5):
    d = _api(lang, action="query", list="search", srsearch=q, srlimit=n, srprop="snippet")
    return [{"title": s["title"], "snippet": _strip_html(s.get("snippet"))} for s in d.get("query", {}).get("search", [])]


def _page_from(d, max_chars):
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    p = pages[0]
    text = clean(p.get("extract", ""))[:max_chars]
    url = p.get("fullurl") or f"https://th.wikipedia.org/wiki/{urllib.parse.quote(p['title'])}"
    return {"title": p["title"], "url": url, "text": text}


def page(title, lang="th", max_chars=20000):
    d = _api(lang, action="query", prop="extracts|info", inprop="url", explaintext="1", redirects="1", titles=title)
    return _page_from(d, max_chars)


def random_page(lang="th", max_chars=20000):
    d = _api(lang, action="query", generator="random", grnnamespace="0", grnlimit="1",
             prop="extracts|info", inprop="url", explaintext="1")
    return _page_from(d, max_chars)
