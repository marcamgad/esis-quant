# Registers the scheduled agents (Stages 1-2). Safe to re-run.
#   ESIS-QUANT Sentinel  hourly 09:00-18:00 Sun-Thu + 21:00 daily  (Agent 1)
#   ESIS-QUANT Corpus    01:00 daily, max 45 min, 1,500 articles    (Agent 4 pretraining text)
# The weekly Saturday fundamentals job already runs the Reference librarian (Agent 3).
#
#   powershell -ExecutionPolicy Bypass -File ops\register_agent_tasks.ps1

$ErrorActionPreference = "Stop"
$root = "D:\Stock\quant_ai"
$pyw = "C:\Users\marc amgad\AppData\Local\Programs\Python\Python311\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "pythonw.exe not found at $pyw" }

# Sentinel
$act = New-ScheduledTaskAction -Execute $pyw -Argument "-m ops.sentinel --alert" -WorkingDirectory $root
$days = "Sunday","Monday","Tuesday","Wednesday","Thursday"
$hourly = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At 9:00am
$hourly.Repetition = (New-ScheduledTaskTrigger -Once -At 9:00am `
    -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Hours 9)).Repetition
$night = New-ScheduledTaskTrigger -Daily -At 9:00pm
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "ESIS-QUANT Sentinel" -Action $act -Trigger @($hourly, $night) `
    -Settings $set -Force -Description "Agent 1: freshness/liveness checks; halts the ticket on critical failures." | Out-Null
Write-Output "registered: ESIS-QUANT Sentinel"

# Corpus collector (no WakeToRun: it is not worth waking the laptop for)
# results-article archive first (earnings study), general corpus once that queue is empty
$act = New-ScheduledTaskAction -Execute $pyw -Argument "-m data.corpus_collector nightly" -WorkingDirectory $root
$trg = New-ScheduledTaskTrigger -Daily -At 1:00am
$set = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 60) `
        -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "ESIS-QUANT Corpus" -Action $act -Trigger $trg -Settings $set -Force `
    -Description "Agent 4 data: polite nightly Arabic article collection for pretraining." | Out-Null
Write-Output "registered: ESIS-QUANT Corpus (01:00 daily)"

# Nightly backup (ops/backup.py): verified zip on D + one copy in OneDrive (drive C,
# synced to the cloud). Wakes the PC, runs on battery: an unprotected ledger is
# worse than a few minutes of battery.
$act = New-ScheduledTaskAction -Execute $pyw -Argument "-m ops.backup" -WorkingDirectory $root
$trg = New-ScheduledTaskTrigger -Daily -At 2:00am
$set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "ESIS-QUANT Backup" -Action $act -Trigger $trg -Settings $set -Force `
    -Description "Nightly verified database backup: D:\Stock\backups + OneDrive." | Out-Null
Write-Output "registered: ESIS-QUANT Backup (02:00 daily)"
