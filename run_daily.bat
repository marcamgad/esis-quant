@echo off
REM ESIS-QUANT daily job launcher (invoked by Windows Task Scheduler).
REM Runs the hardened entry point; all output appended to logs\daily_run.log.
cd /d D:\Stock\quant_ai
"C:\Users\marc amgad\AppData\Local\Programs\Python\Python311\python.exe" -m ops.daily_run >> logs\daily_run.log 2>&1
