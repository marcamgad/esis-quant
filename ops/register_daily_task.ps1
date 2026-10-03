# Registers (or re-registers) the ESIS-QUANT daily job with Windows Task
# Scheduler. EGX trades Sunday-Thursday, so the job runs those days after the
# close. No admin rights required (it's a per-user task).
#
# Usage:   powershell -ExecutionPolicy Bypass -File ops\register_daily_task.ps1
# Remove:  schtasks /delete /tn "ESIS-QUANT Daily" /f

$TaskName = "ESIS-QUANT Daily"
$Bat = "D:\Stock\quant_ai\run_daily.bat"
$RunTime = "17:00"        # local time, after the EGX close (~14:30 Cairo)
$Days = "SUN,MON,TUE,WED,THU"

schtasks /create /tn "$TaskName" /tr "$Bat" /sc weekly /d $Days /st $RunTime /f
if ($LASTEXITCODE -eq 0) {
    # schtasks cannot set these, and its defaults are wrong for a laptop:
    # DisallowStartIfOnBatteries/StopIfGoingOnBatteries both default to TRUE, so
    # the job refuses to start on battery and is KILLED mid-run if the charger is
    # pulled. That is the real cause of the 2026-08-10..15 outage -- the
    # 2026-08-13 17:00 run returned 0xC000013A (terminated) having written only
    # ^C to logs\daily_run.log, so no job_runs row was ever recorded and the
    # failure looked like "the scheduler never fired".
    $t = Get-ScheduledTask -TaskName $TaskName
    $t.Settings.DisallowStartIfOnBatteries = $false
    $t.Settings.StopIfGoingOnBatteries     = $false
    $t.Settings.StartWhenAvailable         = $true   # catch up a missed close
    $t.Settings.RestartCount               = 2
    $t.Settings.RestartInterval            = "PT10M"
    Set-ScheduledTask -TaskName $TaskName -Settings $t.Settings | Out-Null

    Write-Host "Registered '$TaskName' -> runs $Days at $RunTime"
    Write-Host "Battery restrictions cleared; catch-up ON; 2 retries at 10-min spacing."
    schtasks /query /tn "$TaskName" /fo LIST | Select-String "TaskName|Next Run Time|Schedule|Status"
} else {
    Write-Host "Failed to register scheduled task (exit $LASTEXITCODE)"
}
