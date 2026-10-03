@echo off
REM ESIS-QUANT -- record today's ending total.
REM Double-click this at the end of the trading day, type the total your broker
REM shows (stocks + balance), and it stores an exact NAV point and tells you how
REM you are doing. This is the ONE number that is measured, not assumed: the
REM system's own mark drifts from the broker because fills differ from closes
REM and Yahoo's EGX close lands a day late.
setlocal
cd /d "%~dp0"
chcp 65001 >nul

echo ============================================================
echo  ESIS-QUANT  --  end of day total
echo ============================================================
echo.
echo  Open Thndr and add:  stocks value  +  cash balance
echo.
set /p TOTAL=  Total amount now (EGP):

if "%TOTAL%"=="" (
  echo.
  echo  Nothing entered - nothing recorded.
  pause
  exit /b 0
)

set /p NOTE=  Note (optional, press Enter to skip):

echo.
py -3.11 -m ops.portfolio_tracker total %TOTAL% "%NOTE%"

echo.
echo ------------------------------------------------------------
echo  Reminder: one day's number is noise. The measures that mean
echo  something are the weekly review and the go-live gate:
echo     py -3.11 -m ops.status week
echo     py -3.11 -m ops.go_live_gate
echo ------------------------------------------------------------
pause
