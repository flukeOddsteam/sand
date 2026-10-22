"""
เครื่องคำนวณสูตรของแซนด์
- parse():    แปลงสูตรข้อความ เช่น "floor(sqrt(xp / 40)) + 1" เป็นโครงสร้าง (ปลอดภัย ไม่รันโค้ดอื่น)
- evaluate(): คำนวณค่าจากตัวแปร
- render():   วาดสูตรเป็นสัญลักษณ์คณิตศาสตร์ (HTML) เช่น เศษส่วน, √, ⌊ ⌋, ยกกำลัง
"""
import ast
import html
import math
import re

FUNCS = {
    "sqrt": lambda x: math.sqrt(x) if x > 0 else 0.0,
    "log": lambda x: math.log(x) if x > 0 else 0.0,
    "log10": lambda x: math.log10(x) if x > 0 else 0.0,
    "exp": lambda x: math.exp(min(x, 60)),
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
    "tanh": math.tanh,
    "sigmoid": lambda x: 1 / (1 + math.exp(-max(-60, min(60, x)))),
    "sin": math.sin,
    "cos": math.cos,
}
# จำนวนอาร์กิวเมนต์ (ต่ำสุด, สูงสุด)
ARITY = {"min": (2, 9), "max": (2, 9), "clamp": (3, 3), "_if": (3, 3)}
CONSTS = {"pi": math.pi, "e": math.e}

FUNC_HELP = [
    ("sqrt(x)", "รากที่สอง √x"),
    ("x ^ y", "ยกกำลัง"),
    ("floor(x)", "ปัดลง ⌊x⌋"),
    ("ceil(x)", "ปัดขึ้น ⌈x⌉"),
    ("round(x)", "ปัดเศษ"),
    ("abs(x)", "ค่าสัมบูรณ์ |x|"),
    ("log(x)", "ลอการิทึมธรรมชาติ ln x"),
    ("log10(x)", "ลอการิทึมฐาน 10"),
    ("exp(x)", "eˣ"),
    ("min(a, b, …)", "ค่าน้อยสุด"),
    ("max(a, b, …)", "ค่ามากสุด"),
    ("clamp(x, ต่ำ, สูง)", "บีบค่าให้อยู่ในช่วง"),
    ("tanh(x)", "ไฮเพอร์โบลิกแทนเจนต์ (ค่า −1 ถึง 1)"),
    ("sigmoid(x)", "เส้นโค้ง S (ค่า 0 ถึง 1)"),
    ("sin(x), cos(x)", "ตรีโกณมิติ"),
    ("if(เงื่อนไข, a, b)", "ถ้าเงื่อนไขจริงได้ a ไม่งั้นได้ b"),
    ("< <= > >= == !=, and, or", "เปรียบเทียบ (จริง = 1, เท็จ = 0)"),
    ("pi, e", "ค่าคงที่ π และ e"),
    ("a / 0", "หารด้วยศูนย์ได้ 0 (ไม่ error)"),
]


class FormulaError(Exception):
    pass


def normalize(expr):
    for a, b in (("^", "**"), ("×", "*"), ("÷", "/"), ("−", "-"), ("≤", "<="), ("≥", ">="), ("≠", "!=")):
        expr = expr.replace(a, b)
    # "if" เป็นคำสงวนของ Python จึงแปลง if( เป็น _if( ภายใน
    return re.sub(r"\bif\s*\(", "_if(", expr).strip()


_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv)
_CMP = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def _check(n):
    if isinstance(n, ast.Constant):
        if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
            raise FormulaError("ใช้ได้เฉพาะตัวเลข")
    elif isinstance(n, ast.Name):
        pass
    elif isinstance(n, ast.BinOp):
        if not isinstance(n.op, _BIN):
            raise FormulaError("เครื่องหมายนี้ใช้ไม่ได้")
        _check(n.left); _check(n.right)
    elif isinstance(n, ast.UnaryOp):
        if not isinstance(n.op, (ast.USub, ast.UAdd, ast.Not)):
            raise FormulaError("เครื่องหมายนี้ใช้ไม่ได้")
        _check(n.operand)
    elif isinstance(n, ast.Compare):
        if not all(isinstance(o, _CMP) for o in n.ops):
            raise FormulaError("การเปรียบเทียบนี้ใช้ไม่ได้")
        _check(n.left)
        for c in n.comparators:
            _check(c)
    elif isinstance(n, ast.BoolOp):
        for v in n.values:
            _check(v)
    elif isinstance(n, ast.IfExp):
        _check(n.test); _check(n.body); _check(n.orelse)
    elif isinstance(n, ast.Call):
        if not isinstance(n.func, ast.Name) or (n.func.id not in FUNCS and n.func.id != "_if"):
            name = n.func.id if isinstance(n.func, ast.Name) else "?"
            raise FormulaError(f"ไม่รู้จักฟังก์ชัน '{name}'")
        if n.keywords:
            raise FormulaError("ฟังก์ชันไม่รับอาร์กิวเมนต์แบบระบุชื่อ")
        lo, hi = ARITY.get(n.func.id, (1, 1))
        if not lo <= len(n.args) <= hi:
            want = str(lo) if lo == hi else f"{lo}-{hi}"
            raise FormulaError(f"{n.func.id}() ต้องการ {want} ค่า")
        for a in n.args:
            _check(a)
    else:
        raise FormulaError("มีส่วนที่ไม่ใช่สูตรคณิตศาสตร์")


def parse(expr):
    expr = normalize(expr or "")
    if not expr:
        raise FormulaError("สูตรว่าง")
    if len(expr) > 400:
        raise FormulaError("สูตรยาวเกินไป (ไม่เกิน 400 ตัวอักษร)")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise FormulaError("รูปแบบสูตรไม่ถูกต้อง ลองตรวจวงเล็บหรือเครื่องหมาย")
    _check(tree.body)
    return tree.body


def names(node):
    """ตัวแปรทั้งหมดที่สูตรใช้"""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id not in CONSTS and n.id not in FUNCS and n.id != "_if":
            out.add(n.id)
    return out


def evaluate(node, variables):
    def ev(n):
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id in variables and variables[n.id] is not None:
                return float(variables[n.id])
            if n.id in CONSTS:
                return CONSTS[n.id]
            raise FormulaError(f"ไม่รู้จักตัวแปร '{n.id}'")
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            op = type(n.op)
            if op is ast.Add: return a + b
            if op is ast.Sub: return a - b
            if op is ast.Mult: return a * b
            if op is ast.Div: return a / b if b else 0.0
            if op is ast.Mod: return a % b if b else 0.0
            if op is ast.FloorDiv: return float(a // b) if b else 0.0
            if op is ast.Pow:
                if abs(b) > 64 or (a == 0 and b < 0):
                    raise FormulaError("เลขยกกำลังใหญ่เกินไป")
                r = a ** b
                if isinstance(r, complex):
                    raise FormulaError("ผลลัพธ์เป็นจำนวนเชิงซ้อน (ยกกำลังเศษส่วนของเลขติดลบ)")
                return r
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            if isinstance(n.op, ast.USub): return -v
            if isinstance(n.op, ast.Not): return float(not v)
            return v
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                ok = {ast.Lt: left < right, ast.LtE: left <= right, ast.Gt: left > right,
                      ast.GtE: left >= right, ast.Eq: left == right, ast.NotEq: left != right}[type(op)]
                if not ok:
                    return 0.0
                left = right
            return 1.0
        if isinstance(n, ast.BoolOp):
            vals = (ev(v) for v in n.values)
            return float(all(vals)) if isinstance(n.op, ast.And) else float(any(vals))
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.Call):
            if n.func.id == "_if":
                return ev(n.args[1]) if ev(n.args[0]) else ev(n.args[2])
            return float(FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise FormulaError("มีส่วนที่ไม่ใช่สูตรคณิตศาสตร์")

    try:
        r = ev(node)
    except FormulaError:
        raise
    except (OverflowError, ValueError, TypeError, ZeroDivisionError) as e:
        raise FormulaError(f"คำนวณไม่ได้ ({e})")
    if math.isnan(r) or math.isinf(r):
        raise FormulaError("ผลลัพธ์ไม่ใช่ตัวเลขที่ใช้ได้")
    if abs(r) > 1e15:
        raise FormulaError("ผลลัพธ์ใหญ่เกินไป")
    return r


def fmt(x):
    if x is None:
        return "?"
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return f"{int(round(x)):,}"
    if 0 < abs(x) < 0.01:
        return f"{x:.3g}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- render เป็นสัญลักษณ์คณิตศาสตร์
def _prec(n):
    if isinstance(n, ast.IfExp) or (isinstance(n, ast.Call) and n.func.id == "_if"):
        return 10   # วาดเป็นปีกกาแยกกรณี ไม่ต้องใส่วงเล็บ
    if isinstance(n, ast.BoolOp):
        return 1 if isinstance(n.op, ast.Or) else 2
    if isinstance(n, ast.UnaryOp):
        return 3 if isinstance(n.op, ast.Not) else 7
    if isinstance(n, ast.Compare):
        return 4
    if isinstance(n, ast.BinOp):
        op = type(n.op)
        if op in (ast.Add, ast.Sub): return 5
        if op in (ast.Mult, ast.Mod, ast.FloorDiv): return 6
        if op is ast.Pow: return 8
        if op is ast.Div: return 9   # เศษส่วนจัดกลุ่มในตัวเอง
    return 10


_CMP_SYM = {ast.Lt: "&lt;", ast.LtE: "≤", ast.Gt: "&gt;", ast.GtE: "≥", ast.Eq: "=", ast.NotEq: "≠"}


def render(node, vals=None, labels=None):
    """vals=None วาดเป็นชื่อตัวแปร, vals=dict วาดแบบแทนค่าตัวเลข"""
    labels = labels or {}
    E = html.escape

    def op(s):
        return f'<span class="mo">{s}</span>'

    def paren(s):
        return f'<span class="mp">(</span>{s}<span class="mp">)</span>'

    def r(n, parent=0, right=False):
        s = inner(n)
        p = _prec(n)
        if p < parent or (right and p == parent and p in (5, 6)):
            s = paren(s)
        return s

    def cases(cond, a, b):
        return (f'<span class="cases"><span class="brace">{{</span><span class="rows">'
                f'<span>{r(a)}<small>ถ้า</small>{r(cond)}</span>'
                f'<span>{r(b)}<small>กรณีอื่น</small></span></span></span>')

    def inner(n):
        if isinstance(n, ast.Constant):
            return f'<span class="mn">{fmt(float(n.value))}</span>'
        if isinstance(n, ast.Name):
            if n.id == "pi": return '<span class="mn">π</span>'
            if n.id == "e" and n.id not in (vals or {}): return '<span class="mn">e</span>'
            if vals is not None and n.id in vals:
                return f'<span class="mval" title="{E(n.id)}">{fmt(vals[n.id])}</span>'
            return f'<span class="mvar" title="{E(labels.get(n.id, n.id))}">{E(n.id)}</span>'
        if isinstance(n, ast.BinOp):
            t = type(n.op)
            if t is ast.Div:
                return f'<span class="frac"><span>{r(n.left)}</span><span>{r(n.right)}</span></span>'
            if t is ast.Pow:
                return f'{r(n.left, 9)}<sup>{r(n.right)}</sup>'
            sym = {ast.Add: "+", ast.Sub: "−", ast.Mult: "×", ast.Mod: "mod", ast.FloorDiv: "div"}[t]
            p = _prec(n)
            return f'{r(n.left, p)}{op(sym)}{r(n.right, p, True)}'
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.Not):
                return f'{op("ไม่")}{r(n.operand, 4)}'
            return ("−" if isinstance(n.op, ast.USub) else "+") + r(n.operand, 7)
        if isinstance(n, ast.Compare):
            s = r(n.left, 5)
            for o, c in zip(n.ops, n.comparators):
                s += op(_CMP_SYM[type(o)]) + r(c, 5)
            return s
        if isinstance(n, ast.BoolOp):
            word = "และ" if isinstance(n.op, ast.And) else "หรือ"
            return op(word).join(r(v, _prec(n) + 1) for v in n.values)
        if isinstance(n, ast.IfExp):
            return cases(n.test, n.body, n.orelse)
        if isinstance(n, ast.Call):
            f, args = n.func.id, n.args
            if f == "_if":
                return cases(args[0], args[1], args[2])
            if f == "sqrt":
                return ('<span class="sqrt"><svg class="rad" viewBox="0 0 10 24" preserveAspectRatio="none">'
                        '<path d="M0.5 14 L3 12.5 L6 23 L9.6 0.6" vector-effect="non-scaling-stroke"/></svg>'
                        f'<span class="under">{r(args[0])}</span></span>')
            if f == "floor":
                return f'<span class="mb">⌊</span>{r(args[0])}<span class="mb">⌋</span>'
            if f == "ceil":
                return f'<span class="mb">⌈</span>{r(args[0])}<span class="mb">⌉</span>'
            if f == "abs":
                return f'<span class="mb">|</span>{r(args[0])}<span class="mb">|</span>'
            if f == "exp":
                return f'<span class="mn">e</span><sup>{r(args[0])}</sup>'
            name = {"log": "ln", "log10": "log<sub>10</sub>"}.get(f, f)
            return f'<span class="mf">{name}</span>' + paren(", ".join(r(a) for a in args))
        return "?"

    return inner(node)
