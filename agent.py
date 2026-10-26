"""
โหมดช่วยเขียนโค้ดของแซนด์ — เครื่องมือที่สมองโตเรียกใช้ได้ และรั้วความปลอดภัย

วงรอบ (agent loop):
  สมองโตคิด → เขียน <tool_call>{"name": …, "arguments": {…}}</tool_call>
  → แซนด์ตรวจ (อยู่ในโฟลเดอร์งานไหม · ต้องขออนุญาตไหม) → ทำ → ส่งผลกลับใน <tool_response>
  → สมองโตอ่านผลแล้วคิดต่อ จนตอบผู้ใช้โดยไม่เรียกเครื่องมือ = งานเสร็จ

รั้ว:
  · อ่าน/เขียนได้เฉพาะในโฟลเดอร์งานที่ผู้ใช้เลือก (ตรวจ path จริงหลังแก้ symlink แล้ว)
  · เขียนไฟล์ แก้ไฟล์ รันคำสั่ง ต้องให้ผู้ใช้กดอนุญาตก่อนทุกครั้ง (หรือ "อนุญาตทุกครั้งในงานนี้")
  · ก่อนเขียนทับไฟล์ เก็บสำเนาเดิมไว้ใน agent_backups/ ของแซนด์ (นอกโฟลเดอร์งาน)
  · ไม่มีเครื่องมือลบไฟล์ · ห้ามแตะโฟลเดอร์ .git · คำสั่งหมดเวลาใน 120 วินาที
  · เว็บใช้ตัวอ่านเดิมของแซนด์ (เคารพ robots.txt, ไม่เปิดที่อยู่ในเครือข่ายภายใน) และต้องเปิดอินเทอร์เน็ตไว้
"""
import difflib
import json
import os
import platform
import re
import shutil
import subprocess
import time

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".idea", ".vs", "dist", "build", ".next",
             "Library", "Temp", "obj"}
MAX_READ_CHARS = 6000
MAX_OUT_CHARS = 4000
CMD_TIMEOUT = 120
NEEDS_OK = {"write_file", "edit_file", "run_command"}

TOOLS = [
    {"name": "list_dir", "description": "ดูรายชื่อไฟล์และโฟลเดอร์ (ข้ามโฟลเดอร์อย่าง .git node_modules .venv)",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "โฟลเดอร์ เทียบกับโฟลเดอร์งาน (ค่าเริ่มต้น .)"}}}},
    {"name": "read_file", "description": "อ่านไฟล์ข้อความ พร้อมเลขบรรทัด · ไฟล์ยาวให้อ่านทีละช่วงด้วย start_line/end_line",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
         "required": ["path"]}},
    {"name": "search_code", "description": "ค้นข้อความ (regex) ในทุกไฟล์ใต้โฟลเดอร์ บอกไฟล์และเลขบรรทัดที่เจอ",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"}, "path": {"type": "string", "description": "ค่าเริ่มต้น ."}},
         "required": ["pattern"]}},
    {"name": "edit_file", "description": "แก้ไฟล์เดิม: แทนที่ old_text (ต้องตรงทุกตัวอักษรและมีที่เดียวในไฟล์) ด้วย new_text "
                                         "· ใช้ตัวนี้แทน write_file เมื่อแก้ไฟล์ที่มีอยู่แล้ว",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
         "required": ["path", "old_text", "new_text"]}},
    {"name": "write_file", "description": "สร้างไฟล์ใหม่ (หรือเขียนทับทั้งไฟล์) ด้วยเนื้อหาทั้งหมด",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "run_command", "description": "รันคำสั่งในโฟลเดอร์งาน เช่น รันโปรแกรมหรือเทสต์ ได้ผลลัพธ์และ exit code "
                                           f"(หมดเวลาใน {CMD_TIMEOUT} วินาที ห้ามรันโปรแกรมที่ค้างรอตลอด เช่น เซิร์ฟเวอร์)",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "web_search", "description": "ค้นในอินเทอร์เน็ต ได้ชื่อเว็บ ลิงก์ และข้อความย่อ",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "read_url", "description": "เปิดอ่านหน้าเว็บ (เช่น เอกสารของไลบรารี) ได้เฉพาะเนื้อความ",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
]
TOOL_NAMES = {t["name"] for t in TOOLS}


class ToolError(Exception):
    pass


def check_root(path):
    """ตรวจโฟลเดอร์งานที่ผู้ใช้เลือก → คืน path จริง หรือข้อความบอกว่าทำไมใช้ไม่ได้"""
    if not path or not path.strip():
        return None, "ยังไม่ได้เลือกโฟลเดอร์งาน"
    p = os.path.realpath(os.path.expanduser(path.strip().strip('"')))
    if not os.path.isdir(p):
        return None, f"ไม่พบโฟลเดอร์ {p}"
    home = os.path.realpath(os.path.expanduser("~"))
    if p == os.path.dirname(p) or p == home:
        return None, "เลือกทั้งไดรฟ์หรือทั้งโฟลเดอร์บ้านไม่ได้ (กว้างเกินไป) — เลือกโฟลเดอร์ของโปรเจกต์"
    return p, None


class Workspace:
    def __init__(self, root, backup_dir):
        self.root = root
        self.backup_dir = backup_dir

    def path(self, rel, write=False):
        rel = (rel or ".").strip().strip('"')
        p = os.path.realpath(os.path.join(self.root, os.path.expanduser(rel)))
        if os.path.commonpath([p, self.root]) != self.root:
            raise ToolError(f"“{rel}” อยู่นอกโฟลเดอร์งาน — ใช้ได้เฉพาะไฟล์ใน {self.root}")
        if write and ".git" in os.path.relpath(p, self.root).split(os.sep):
            raise ToolError("ห้ามแก้ไฟล์ในโฟลเดอร์ .git")
        return p

    def rel(self, p):
        return os.path.relpath(p, self.root).replace(os.sep, "/")

    def backup(self, p):
        if os.path.exists(p):
            dest = os.path.join(self.backup_dir, time.strftime("%Y%m%d-%H%M%S"), self.rel(p))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(p, dest)
            return dest
        return None


def _read_text(p):
    if os.path.getsize(p) > 2_000_000:
        raise ToolError("ไฟล์ใหญ่เกิน 2 MB")
    with open(p, "rb") as f:
        raw = f.read()
    if b"\0" in raw[:4096]:
        raise ToolError("เป็นไฟล์ไบนารี (ไม่ใช่ข้อความ)")
    return raw.decode("utf-8", "replace")


def _cut(s, n):
    return s if len(s) <= n else s[:n] + f"\n… (ตัดเหลือ {n:,} ตัวอักษรจาก {len(s):,})"


def diff_preview(old, new, name, limit=3000):
    d = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"เดิม/{name}", f"ใหม่/{name}", n=2))
    return _cut(d or "(เนื้อหาเหมือนเดิม)", limit)


def preview(ws, name, a):
    """สิ่งที่แสดงให้ผู้ใช้ดูก่อนกดอนุญาต"""
    if name == "run_command":
        return {"title": "รันคำสั่ง", "body": a.get("command", "")}
    p = ws.path(a.get("path"), write=True)
    old = _read_text(p) if os.path.exists(p) else ""
    if name == "write_file":
        new = a.get("content", "")
        title = ("เขียนทับ " if os.path.exists(p) else "สร้างไฟล์ ") + ws.rel(p)
    else:
        if a.get("old_text", "") not in old:
            raise ToolError("ไม่เจอ old_text ในไฟล์ — อ่านไฟล์อีกครั้งแล้วคัดลอกข้อความให้ตรงทุกตัวอักษร")
        new = old.replace(a["old_text"], a.get("new_text", ""), 1)
        title = "แก้ไฟล์ " + ws.rel(p)
    return {"title": title, "body": diff_preview(old, new, ws.rel(p)), "diff": True}


def run_tool(ws, name, a, net=None):
    """ทำตามที่สมองโตสั่ง → (ข้อความที่ส่งกลับให้สมองโตอ่าน, สรุปสั้นๆ ให้ผู้ใช้ดู)"""
    if name == "list_dir":
        p = ws.path(a.get("path"))
        if not os.path.isdir(p):
            raise ToolError("ไม่ใช่โฟลเดอร์")
        rows = []
        for e in sorted(os.scandir(p), key=lambda e: (not e.is_dir(), e.name.lower())):
            if e.name in SKIP_DIRS:
                continue
            rows.append(e.name + "/" if e.is_dir() else f"{e.name}  ({e.stat().st_size:,} ไบต์)")
            if len(rows) >= 200:
                rows.append("… (มีมากกว่านี้)")
                break
        return "\n".join(rows) or "(โฟลเดอร์ว่าง)", f"ดู {ws.rel(p)}/ · {len(rows)} รายการ"
    if name == "read_file":
        p = ws.path(a.get("path"))
        if not os.path.isfile(p):
            raise ToolError("ไม่พบไฟล์")
        lines = _read_text(p).splitlines()
        s = max(1, int(a.get("start_line") or 1))
        e = min(len(lines), int(a.get("end_line") or s + 249))
        out, used = [], 0
        for i in range(s - 1, e):
            row = f"{i + 1:>5}| {lines[i]}"
            if used + len(row) > MAX_READ_CHARS:
                e = i
                break
            out.append(row)
            used += len(row) + 1
        more = f"\n(ไฟล์มี {len(lines)} บรรทัด · แสดง {s}–{e} · อ่านต่อด้วย start_line={e + 1})" if e < len(lines) else ""
        return "\n".join(out) + more, f"อ่าน {ws.rel(p)} บรรทัด {s}–{e} จาก {len(lines)}"
    if name == "search_code":
        try:
            rx = re.compile(a.get("pattern", ""))
        except re.error:
            rx = re.compile(re.escape(a.get("pattern", "")))
        base = ws.path(a.get("path"))
        hits, files = [], 0
        for d, dirs, names in os.walk(base):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for n in names:
                fp = os.path.join(d, n)
                try:
                    if os.path.getsize(fp) > 1_000_000:
                        continue
                    text = _read_text(fp)
                except (ToolError, OSError):
                    continue
                files += 1
                for i, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{ws.rel(fp)}:{i}: {line.strip()[:160]}")
                        if len(hits) >= 60:
                            break
                if len(hits) >= 60:
                    break
            if len(hits) >= 60:
                break
        return "\n".join(hits) or "ไม่เจอ", f"ค้น “{a.get('pattern', '')}” เจอ {len(hits)} ที่ (ดู {files} ไฟล์)"
    if name in ("write_file", "edit_file"):
        p = ws.path(a.get("path"), write=True)
        old = _read_text(p) if os.path.exists(p) else None
        if name == "edit_file":
            if old is None:
                raise ToolError("ไม่พบไฟล์ — ถ้าจะสร้างใหม่ใช้ write_file")
            n = old.count(a.get("old_text", ""))
            if not a.get("old_text") or n == 0:
                raise ToolError("ไม่เจอ old_text ในไฟล์ — อ่านไฟล์อีกครั้งแล้วคัดลอกข้อความให้ตรงทุกตัวอักษร")
            if n > 1:
                raise ToolError(f"old_text มี {n} ที่ในไฟล์ — ใส่บรรทัดรอบๆ เพิ่มให้เหลือที่เดียว")
            new = old.replace(a["old_text"], a.get("new_text", ""), 1)
        else:
            new = a.get("content", "")
        bk = ws.backup(p)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(new)
        verb = "แก้" if name == "edit_file" else ("เขียนทับ" if old is not None else "สร้าง")
        return (f"{verb} {ws.rel(p)} แล้ว ({len(new.splitlines())} บรรทัด)",
                f"{verb} {ws.rel(p)}" + (" · เก็บสำเนาเดิมไว้แล้ว" if bk else ""))
    if name == "run_command":
        cmd = a.get("command", "").strip()
        if not cmd:
            raise ToolError("ไม่มีคำสั่ง")
        t0 = time.time()
        try:
            r = subprocess.run(cmd, shell=True, cwd=ws.root, capture_output=True, timeout=CMD_TIMEOUT,
                               stdin=subprocess.DEVNULL)
            code = r.returncode
            out = (r.stdout + (b"\n" + r.stderr if r.stderr else b"")).decode("utf-8", "replace")
        except subprocess.TimeoutExpired as e:
            code = "หมดเวลา"
            out = ((e.stdout or b"") + (e.stderr or b"")).decode("utf-8", "replace") + f"\n(หยุดเพราะเกิน {CMD_TIMEOUT} วินาที)"
        out = out.strip()
        if len(out) > MAX_OUT_CHARS:                 # เก็บท้ายไว้ เพราะ error มักอยู่ท้าย
            out = f"… (ตัดต้นออก เหลือ {MAX_OUT_CHARS:,} ตัวอักษรท้าย)\n" + out[-MAX_OUT_CHARS:]
        return f"exit code: {code}\n{out or '(ไม่มีผลลัพธ์)'}", f"รัน `{cmd[:80]}` → exit {code} · {time.time() - t0:.1f} วินาที"
    if name in ("web_search", "read_url"):
        if not net or not net.get("on"):
            raise ToolError("ผู้ใช้ยังไม่ได้เปิดให้แซนด์ใช้อินเทอร์เน็ต (แท็บ 📚 ข้อมูลฝึก) — ตอบจากความรู้ที่มี")
        import web as W
        if name == "web_search":
            q = a.get("query", "").strip()
            res = W.web_search(q, 6, net.get("provider", "ddg"), net.get("key", ""))
            text = "\n\n".join(f"[{i + 1}] {r['title']}\n{r['url']}\n{r['snippet']}" for i, r in enumerate(res))
            return text or "ไม่เจอผลค้น", f"ค้นเว็บ “{q}” เจอ {len(res)} ผล"
        url = a.get("url", "").strip()
        if not url.startswith(("http://", "https://")):
            raise ToolError("ต้องเป็นลิงก์ http(s)")
        pg = W.fetch_page(url, max_chars=MAX_OUT_CHARS)
        return f"{pg['title']}\n{pg['url']}\n\n{pg['text']}", f"อ่านเว็บ {pg['title'][:60]}"
    raise ToolError(f"ไม่มีเครื่องมือชื่อ {name} — ใช้ได้: {', '.join(sorted(TOOL_NAMES))}")


def parse_call(text):
    """ดึง {"name": …, "arguments": {…}} จากระหว่าง <tool_call> … </tool_call>"""
    m = re.search(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S)
    body = (m.group(1) if m else text).strip()
    body = re.sub(r"^```(?:json)?|```$", "", body).strip()
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        s, e = body.find("{"), body.rfind("}")
        try:
            d = json.loads(body[s:e + 1]) if s >= 0 < e else None
        except json.JSONDecodeError:
            d = None
    if not isinstance(d, dict) or "name" not in d:
        raise ToolError('รูปแบบ tool_call ไม่ถูก ต้องเป็น JSON บรรทัดเดียว: {"name": "ชื่อ", "arguments": {...}}')
    args = d.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            raise ToolError("arguments ต้องเป็น JSON object")
    return d["name"], args


def system_prompt(root, net_on):
    osname = {"Darwin": "macOS", "Windows": "Windows (cmd)"}.get(platform.system(), platform.system())
    tools = "\n".join(json.dumps({"type": "function", "function": t}, ensure_ascii=False) for t in TOOLS)
    return (
        "คุณคือ \"แซนด์\" (Sand) ผู้ช่วยเขียนโค้ดที่ทำงานบนคอมพิวเตอร์ของผู้ใช้\n"
        f"โฟลเดอร์งาน: {root}\nระบบปฏิบัติการ: {osname} (คำสั่งที่รันต้องใช้ได้กับระบบนี้)\n"
        f"อินเทอร์เน็ต: {'เปิดอยู่ ใช้ web_search / read_url ได้' if net_on else 'ปิดอยู่ ห้ามใช้ web_search / read_url'}\n\n"
        "วิธีทำงาน:\n"
        "- ดูก่อนลงมือ: ใช้ list_dir, search_code, read_file ให้เข้าใจโค้ดจริงก่อน ห้ามเดาเนื้อหาไฟล์\n"
        "- แก้ไฟล์เดิมด้วย edit_file ทีละจุด (old_text คัดลอกจากที่อ่านมาให้ตรงทุกตัวอักษร ไม่รวมเลขบรรทัด) "
        "ใช้ write_file เฉพาะไฟล์ใหม่\n"
        "- แก้เสร็จแล้วรันทดสอบด้วย run_command ถ้าทำได้ ถ้าเจอ error ให้อ่าน error แล้วแก้ต่อ\n"
        "- เรียกเครื่องมือครั้งละ 1 ตัว แล้วรอผล · ผู้ใช้อาจไม่อนุญาตบางอย่าง ให้เคารพและหาทางอื่นหรือถามผู้ใช้\n"
        "- ไม่รู้หรือต้องใช้ข้อมูลล่าสุด (เวอร์ชันไลบรารี เอกสาร API) ให้ค้นเว็บ อย่าแต่งขึ้นเอง\n"
        "- จบงานด้วยการสรุปสั้นๆ เป็นภาษาเดียวกับผู้ใช้ ว่าแก้อะไร ไฟล์ไหน ทดสอบแล้วได้ผลอย่างไร\n\n"
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        f"You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    )
