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


# ================================================================ ค้นทั้งเว็บ + อ่านเว็บทั่วไป
import ipaddress
import socket
import urllib.robotparser
from html.parser import HTMLParser

DDG_URL = os.environ.get("SAND_DDG_URL", "https://html.duckduckgo.com/html/")
TAVILY_URL = os.environ.get("SAND_TAVILY_URL", "https://api.tavily.com/search")
ALLOW_LOCAL = os.environ.get("SAND_ALLOW_LOCAL") == "1"      # ใช้ตอนทดสอบเท่านั้น
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) SandAI/1.0 (+https://github.com/flukeOddsteam/sand)"
_domain_last = {}
_robots = {}
_ddg_last = [0.0]


class BlockedURL(Exception):
    pass


def _check_host(url):
    """กันไม่ให้ลิงก์จากผลค้นพาเข้าเครื่องในบ้าน/เครือข่ายส่วนตัว"""
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise BlockedURL("รองรับเฉพาะลิงก์ http/https")
    if ALLOW_LOCAL:
        return
    for info in socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80)):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise BlockedURL(f"ไม่อ่านที่อยู่ภายในเครือข่าย ({u.hostname})")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirect)


def _polite(domain, gap=2.0):
    with _lock:
        wait = gap - (time.time() - _domain_last.get(domain, 0))
        _domain_last[domain] = time.time() + max(0, wait)
    if wait > 0:
        time.sleep(wait)


def _http(url, data=None, headers=None, limit=3_000_000, timeout=20):
    _check_host(url)
    h = {"User-Agent": BROWSER_UA, "Accept-Language": "th,en;q=0.8"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with _opener.open(req, timeout=timeout) as r:
        body = r.read(limit + 1)
        return r, body[:limit]


def allowed_by_robots(url):
    u = urllib.parse.urlparse(url)
    base = f"{u.scheme}://{u.netloc}"
    rp = _robots.get(base)
    if rp is None or time.time() - rp[1] > 3600:
        parser = urllib.robotparser.RobotFileParser()
        try:
            _, body = _http(base + "/robots.txt", limit=500_000, timeout=10)
            parser.parse(body.decode("utf-8", "replace").splitlines())
        except Exception:
            parser.parse([])                     # ไม่มี robots.txt = อนุญาต
        rp = (parser, time.time())
        _robots[base] = rp
    return rp[0].can_fetch("SandAI", url)


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe", "button", "select"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "article", "section", "tr", "blockquote", "pre", "td"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip, self.title, self.in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif tag == "meta":
            a = dict(attrs)
            if a.get("property") == "og:title" and a.get("content"):
                self.title = a["content"]
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self.in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title and not self.title:
            self.title = data.strip()
        elif not self.skip:
            self.parts.append(data)


BOILER = re.compile(r"(cookie|คุกกี้|สงวนลิขสิทธิ์|copyright|©|all rights reserved|สมัครสมาชิก|เข้าสู่ระบบ|subscribe|privacy policy|นโยบายความเป็นส่วนตัว)", re.I)


def _decode(body, ctype):
    m = re.search(r"charset=([\w-]+)", ctype or "", re.I) or re.search(rb'charset=["\']?([\w-]+)', body[:4000], re.I)
    enc = m.group(1) if m else "utf-8"
    enc = enc.decode() if isinstance(enc, bytes) else enc
    for e in (enc, "utf-8", "cp874"):
        try:
            return body.decode(e)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


def fetch_page(url, max_chars=20000):
    """เปิดอ่านเว็บทั่วไป: เคารพ robots.txt, เว้นระยะต่อโดเมน, เอาเฉพาะเนื้อความ"""
    if not allowed_by_robots(url):
        raise BlockedURL("เว็บนี้ไม่อนุญาตให้บอทอ่าน (robots.txt)")
    _polite(urllib.parse.urlparse(url).netloc)
    r, body = _http(url)
    ctype = r.headers.get("Content-Type", "")
    if "html" not in ctype and "text/plain" not in ctype:
        raise BlockedURL(f"ไม่ใช่หน้าเว็บข้อความ ({ctype.split(';')[0]})")
    text = _decode(body, ctype)
    if "text/plain" in ctype:
        title, lines = url, text.splitlines()
    else:
        ex = _TextExtractor()
        ex.feed(text)
        title = ex.title or url
        lines = "".join(ex.parts).splitlines()
    keep, seen = [], set()
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 25 or line in seen or (len(line) < 120 and BOILER.search(line)):
            continue
        seen.add(line)
        keep.append(line)
    return {"title": html.unescape(title.strip())[:200], "url": r.geturl() if hasattr(r, "geturl") else url,
            "text": "\n".join(keep)[:max_chars]}


def _ddg(q, n):
    with _lock:
        wait = 3.0 - (time.time() - _ddg_last[0])
        _ddg_last[0] = time.time() + max(0, wait)
    if wait > 0:
        time.sleep(wait)
    data = urllib.parse.urlencode({"q": q, "kl": "th-th"}).encode()
    _, body = _http(DDG_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    page_html = body.decode("utf-8", "replace")
    out = []
    anchors = list(re.finditer(r'<a\b([^>]*\bclass="[^"]*\bresult__a\b[^"]*"[^>]*)>(.*?)</a>', page_html, re.S))
    for i, m in enumerate(anchors):
        attrs, title = m.group(1), m.group(2)
        rest = page_html[m.end():anchors[i + 1].start() if i + 1 < len(anchors) else len(page_html)]
        hm = re.search(r'href="([^"]+)"', attrs)
        if not hm:
            continue
        href = html.unescape(hm.group(1))
        if "uddg=" in href:
            href = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [""])[0]
        elif href.startswith("//"):
            href = "https:" + href
        if not href.startswith("http") or "duckduckgo.com/y.js" in href:
            continue                                        # ข้ามโฆษณา
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', rest, re.S)
        out.append({"title": _strip_html(title), "url": href, "snippet": _strip_html(sn.group(1)) if sn else ""})
        if len(out) >= n:
            break
    if not out and ("anomaly" in page_html.lower() or "captcha" in page_html.lower()):
        raise BlockedURL("DuckDuckGo ขอให้ช้าลง (ค้นถี่เกินไป) — ลองใหม่ภายหลัง หรือใช้ Tavily")
    return out


def _tavily(q, n, key):
    body = json.dumps({"query": q, "max_results": n, "search_depth": "basic"}).encode()
    _, data = _http(TAVILY_URL, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return [{"title": r.get("title", ""), "url": r["url"], "snippet": r.get("content", "")[:300]}
            for r in json.loads(data).get("results", [])]


def web_search(q, n=6, provider="ddg", key=""):
    if provider == "tavily":
        if not key:
            raise BlockedURL("ยังไม่ได้ใส่ Tavily API key")
        return _tavily(q, n, key)
    return _ddg(q, n)


# ================================================================ ชุดข้อมูลบทสนทนาจาก Hugging Face (datasets-server API)
HF_DS = os.environ.get("SAND_HF_DS", "https://datasets-server.huggingface.co")


def _hf(path, **params):
    url = HF_DS + path + "?" + urllib.parse.urlencode(params)
    _polite(urllib.parse.urlparse(HF_DS).netloc, gap=1.0)
    _, body = _http(url, headers={"Accept": "application/json", "User-Agent": UA}, timeout=40)
    return json.loads(body)


def hf_splits(dataset):
    return [(s["config"], s["split"]) for s in _hf("/splits", dataset=dataset).get("splits", [])]


def hf_filter(dataset, config, split, where, offset=0, length=100):
    """แถวที่ตรงเงื่อนไข (เงื่อนไขแบบ SQL เช่น "language_code"='tha') ครั้งละไม่เกิน 100 แถว"""
    d = _hf("/filter", dataset=dataset, config=config, split=split, where=where, offset=offset, length=min(100, length))
    return [r["row"] for r in d.get("rows", [])], d.get("num_rows_total", 0)
