#!/usr/bin/env bash
# เปิดแซนด์ (Mac / Linux) — ดับเบิลคลิก หรือรัน ./start-mac.command
cd "$(dirname "$0")"

OS="$(uname -s)"
HW="$(uname -m)"

# Python ที่ใช้ได้: 3.10–3.14 และ (บน Mac ชิป M) ต้องเป็นรุ่น arm64 ไม่ใช่รุ่น Intel ที่รันผ่าน Rosetta
py_ok() {
  "$1" - "$OS" "$HW" <<'PY' 2>/dev/null
import sys, platform
os_, hw = sys.argv[1], sys.argv[2]
ok = (3, 10) <= sys.version_info[:2] <= (3, 14)
if os_ == "Darwin" and hw == "arm64":
    ok = ok and platform.machine() == "arm64"
sys.exit(0 if ok else 1)
PY
}

describe() {
  "$1" -c "import sys,platform;print(f'{sys.version.split()[0]} ({platform.machine()})')" 2>/dev/null || echo "เปิดไม่ได้"
}

# ถ้า .venv เดิมสร้างจาก Python ที่ใช้ไม่ได้ ให้ลบทิ้งแล้วสร้างใหม่
if [ -x .venv/bin/python ] && ! py_ok .venv/bin/python; then
  echo "สภาพแวดล้อมเดิมใช้ Python $(describe .venv/bin/python) ซึ่ง PyTorch ไม่รองรับ — กำลังสร้างใหม่"
  rm -rf .venv
fi

if [ ! -x .venv/bin/python ]; then
  PY=""
  CANDIDATES="python3.13 python3.12 python3.11 python3.10 python3.14 python3
    /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
  for c in $CANDIDATES; do
    if command -v "$c" >/dev/null 2>&1 && py_ok "$c"; then PY="$c"; break; fi
  done

  if [ -z "$PY" ]; then
    echo ""
    echo "❌ ไม่พบ Python ที่ PyTorch รองรับ (ต้องเป็น 3.10–3.14$( [ "$OS" = Darwin ] && [ "$HW" = arm64 ] && echo ' รุ่น arm64'))"
    echo "   Python ที่เจอในเครื่อง:"
    for c in python3 python3.9 python3.10 python3.11 python3.12 python3.13 python3.14 python3.15; do
      command -v "$c" >/dev/null 2>&1 && echo "     $c → $(describe "$c")  [$(command -v "$c")]"
    done
    if [ "$OS" = Darwin ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = "1" ]; then
      echo "   ⚠ Terminal นี้กำลังรันผ่าน Rosetta (โหมด Intel) — ปิด 'Open using Rosetta' ของแอป Terminal ก่อน"
    fi
    echo ""
    echo "   วิธีแก้: ติดตั้ง Python 3.12 จาก https://www.python.org/downloads/macos/"
    echo "   (เลือก macOS 64-bit universal2 installer) แล้วรันไฟล์นี้อีกครั้ง"
    [ "$OS" = Darwin ] && sw_vers -productVersion | awk -F. '$1<14{print "   ⚠ macOS ของคุณเวอร์ชัน "$0" — PyTorch รุ่นใหม่ต้องการ macOS 14 ขึ้นไป"}'
    read -r -p "กด Enter เพื่อปิด" _
    exit 1
  fi

  echo "ใช้ $PY → Python $(describe "$PY")"
  echo "ครั้งแรก: กำลังเตรียมสภาพแวดล้อม..."
  "$PY" -m venv .venv || { echo "สร้าง .venv ไม่สำเร็จ"; read -r _; exit 1; }
fi

source .venv/bin/activate
if ! python -c "import torch" 2>/dev/null; then
  echo "กำลังติดตั้ง PyTorch (ครั้งแรกครั้งเดียว ~200MB)..."
  python -m pip install --upgrade pip >/dev/null
  if ! python -m pip install torch; then
    echo ""
    echo "❌ ติดตั้ง PyTorch ไม่สำเร็จ · Python ใน .venv: $(describe python) · macOS: $(sw_vers -productVersion 2>/dev/null || uname -r)"
    echo "   ส่งข้อความนี้ให้ Claude ดูได้เลย"
    rm -rf .venv
    read -r -p "กด Enter เพื่อปิด" _
    exit 1
  fi
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
python app.py
