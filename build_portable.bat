@echo off
setlocal

python -m PyInstaller ^
  --onefile ^
  --windowed ^
  --clean ^
  --name "SVDImageCompressor" ^
  main.py

echo.
echo Build selesai.
echo File portable ada di: dist\SVDImageCompressor.exe
endlocal
