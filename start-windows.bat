@echo off
chcp 65001 >nul
setlocal
REM เปิดแซนด์ (Windows) — ดับเบิลคลิก หรือพิมพ์ start-windows.bat ใน cmd
cd /d "%~dp0"

REM Python ที่ใช้ได้: 3.10-3.14 รุ่น 64-bit (PyTorch ไม่มีให้รุ่น 32-bit หรือเวอร์ชันอื่น)
set "CHECK=import sys,struct;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,14) and struct.calcsize('P')==8 else 1)"
set "DESC=import sys,platform;print(sys.version.split()[0], platform.architecture()[0])"

REM ถ้า .venv เดิมสร้างจาก Python ที่ใช้ไม่ได้ ให้ลบทิ้งแล้วสร้างใหม่
if not exist ".venv\Scripts\python.exe" goto :makevenv
".venv\Scripts\python.exe" -c "%CHECK%" >nul 2>&1 && goto :haveenv
echo สภาพแวดล้อมเดิมใช้ Python ที่ PyTorch ไม่รองรับ — กำลังสร้างใหม่
rmdir /s /q .venv

:makevenv
set "PY="
REM ลองผ่านตัวเรียก py ก่อน (ติดตั้งมากับ Python จาก python.org) เลือก 3.12 เป็นอันดับแรก
for %%V in (3.12 3.13 3.11 3.10 3.14) do (
  if not defined PY (
    py -%%V-64 -c "%CHECK%" >nul 2>&1 && set "PY=py -%%V-64"
  )
)
if not defined PY (
  python -c "%CHECK%" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  python3 -c "%CHECK%" >nul 2>&1 && set "PY=python3"
)

if not defined PY goto :nopython

echo ใช้ %PY%
%PY% -c "%DESC%"
echo ครั้งแรก: กำลังเตรียมสภาพแวดล้อม...
%PY% -m venv .venv
if errorlevel 1 (
  echo สร้าง .venv ไม่สำเร็จ
  pause
  exit /b 1
)

:haveenv
call ".venv\Scripts\activate.bat"
python -c "import torch, tokenizers, numpy" >nul 2>&1 && goto :run

echo กำลังติดตั้ง PyTorch และ tokenizers (ครั้งแรกครั้งเดียว)...
python -m pip install --upgrade pip >nul
python -m pip install tokenizers numpy
if errorlevel 1 goto :installfail

REM มีการ์ดจอ NVIDIA → ลองรุ่นที่ใช้ GPU ก่อน ไม่ได้ก็ใช้รุ่น CPU
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :cputorch
echo พบการ์ดจอ NVIDIA — ลองติดตั้ง PyTorch รุ่น GPU (~2.5GB)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 && goto :run
echo ติดตั้งรุ่น GPU ไม่สำเร็จ — ใช้รุ่น CPU แทน

:cputorch
python -m pip install torch
if errorlevel 1 goto :installfail

:run
set PYTHONUTF8=1
python app.py
pause
exit /b 0

:installfail
echo.
echo ❌ ติดตั้ง PyTorch ไม่สำเร็จ
python -c "%DESC%"
echo    ส่งข้อความนี้ให้ Claude ดูได้เลย
call ".venv\Scripts\deactivate.bat" >nul 2>&1
rmdir /s /q .venv
pause
exit /b 1

:nopython
echo.
echo ❌ ไม่พบ Python ที่ PyTorch รองรับ (ต้องเป็น 3.10-3.14 รุ่น 64-bit)
echo    Python ที่เจอในเครื่อง:
py -0p 2>nul
for %%C in (python python3) do (
  where %%C >nul 2>&1 && (
    echo      %%C:
    %%C -c "%DESC%" 2>nul || echo        เปิดไม่ได้ (อาจเป็นทางลัดไป Microsoft Store^)
  )
)
echo.
echo    วิธีแก้: ติดตั้ง Python 3.12 จาก https://www.python.org/downloads/windows/
echo    เลือก "Windows installer (64-bit)" และติ๊ก "Add python.exe to PATH"
echo    แล้วปิดหน้าต่างนี้ เปิด cmd ใหม่ และรันไฟล์นี้อีกครั้ง
pause
exit /b 1
