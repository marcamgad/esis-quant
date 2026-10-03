# Registers the weekly fundamentals job and hardens every ESIS-QUANT task
# against a sleeping laptop.
#
# WHY. Between 2026-08-25 and 2026-09-10 the daily job succeeded once. Battery
# settings were already correct; the remaining failures were a machine that was
# asleep or shut at run time (0xC000013A). Three settings address that:
#   WakeToRun          -- wake the machine from sleep to run the task. Needs
#                         "Allow wake timers" enabled in the active power plan.
#   StartWhenAvailable -- if the run time was missed, run as soon as possible.
#   RestartCount/Interval -- retry after a kill instead of waiting a day.
#
# Run from an elevated or normal PowerShell:
#   powershell -ExecutionPolicy Bypass -File ops\register_resilient_tasks.ps1

$ErrorActionPreference = "Stop"
$root = "D:\Stock\quant_ai"
$py = (Get-Command py).Source

# Weekly fundamentals: Saturday 11:00. EGX is closed Fri/Sat, so it never
# competes with the trading-day jobs, and a missed run catches up on wake.
$act = New-ScheduledTaskAction -Execute $py -Argument "-3.11 -m ops.fundamentals_refresh" -WorkingDirectory $root
$trg = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 11:00am
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -WakeToRun -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 30) `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName "ESIS-QUANT Fundamentals Weekly" -Action $act -Trigger $trg `
    -Settings $set -Force `
    -Description "Weekly security master + fundamentals refresh (~25 min). Separate from daily jobs." | Out-Null
Write-Output "registered: ESIS-QUANT Fundamentals Weekly (Sat 11:00)"

# Harden every existing ESIS task.
Get-ScheduledTask | Where-Object { $_.TaskName -like "ESIS-QUANT*" } | ForEach-Object {
    $s = $_.Settings
    $s.WakeToRun = $true
    $s.StartWhenAvailable = $true
    $s.DisallowStartIfOnBatteries = $false
    $s.StopIfGoingOnBatteries = $false
    if ($s.RestartCount -lt 2) { $s.RestartCount = 2; $s.RestartInterval = "PT15M" }
    Set-ScheduledTask -TaskName $_.TaskName -Settings $s | Out-Null
    Write-Output ("hardened: {0,-32} wake={1} catchup={2} restarts={3}" -f `
        $_.TaskName, $s.WakeToRun, $s.StartWhenAvailable, $s.RestartCount)
}

# Report whether wake timers are actually allowed -- WakeToRun is inert without it.
$ac = powercfg /query SCHEME_CURRENT SUB_SLEEP RTCWAKE 2>$null | Select-String "Current AC Power Setting Index"
$dc = powercfg /query SCHEME_CURRENT SUB_SLEEP RTCWAKE 2>$null | Select-String "Current DC Power Setting Index"
Write-Output "wake timers (0=disabled,1=enabled,2=important only): AC $ac | DC $dc"
