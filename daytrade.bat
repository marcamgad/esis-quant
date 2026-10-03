@echo off
REM ESIS-QUANT -- daily DAY-TRADE buy list.
REM Double-click each morning. Writes orders\daytrade_<today>.csv sized to your
REM capital, prints it, and opens the file so you can fill in your real fills.
REM Sell everything before the 2:30 close, then run record_total.bat / the log
REM command it prints at the end.
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set CAPITAL=1000
if not "%~1"=="" set CAPITAL=%~1

echo ============================================================
echo  ESIS-QUANT  --  day-trade buy list  (%CAPITAL% EGP)
echo ============================================================
echo.

REM Refresh the signal from stored data only (no fetch) so this is instant.
REM If the daily job has not run today the signal may be a day old - the CSV
REM header always states the as-of date it was built from.
py -3.11 -m analysis.day_trade ticket %CAPITAL%
if errorlevel 1 (
  echo.
  echo  Could not build the list. Most likely the daily job has not run yet:
  echo      py -3.11 -m ops.daily_run
  pause
  exit /b 1
)

echo.
echo ------------------------------------------------------------
echo  Opening the CSV. Fill FILLED_shares / FILLED_price as you trade.
echo  SELL EVERYTHING before 14:30, then log the day with:
echo      py -3.11 -m analysis.day_trade log ^<buy_total^> ^<sell_total^>
echo ------------------------------------------------------------

for /f "delims=" %%F in ('powershell -NoProfile -Command "Get-ChildItem '%~dp0orders\daytrade_*.csv' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set TICKET=%%F
if exist "%TICKET%" start "" "%TICKET%"

echo.
py -3.11 -m analysis.day_trade
echo.
pause
