@echo off
chcp 936 >nul
cd /d "%~dp0"

rem ---- 服务已在运行时：直接开浏览器，不重复占内存 ----
netstat -ano | findstr "LISTENING" | findstr ":7860" >nul 2>&1
if not errorlevel 1 goto :already

echo ============================================================
echo   RAG 智能问答 - 本地服务
echo ============================================================
echo.
echo   正在加载索引（向量 331 万条 + BM25 + faiss 13.6 GB）
echo   首次约需 1 分钟，请勿关闭本窗口
echo.
echo   加载完成后浏览器会自动打开：
echo   http://127.0.0.1:7860
echo.
echo   停止服务：直接关闭本窗口
echo ============================================================
echo.

"C:\Users\搞懂\AppData\Local\Programs\Python\Python313\python.exe" -u app\gradio_app.py --port 7860

echo.
echo   服务已停止，按任意键关闭窗口
pause >nul
exit /b

:already
echo ============================================================
echo   服务已经在运行，直接打开浏览器
echo   http://127.0.0.1:7860
echo ============================================================
start "" http://127.0.0.1:7860
timeout /t 2 >nul
exit /b
