# ESIS-QUANT -- daily interactive popup.
# Shows today's report (suggestion + cost/capacity + profit estimate + news),
# then asks what you did and records your decision. Nothing here places a real
# order -- you trade at your broker, then tell the tracker what you did.
#
# Run manually:   powershell -ExecutionPolicy Bypass -File daily.ps1
# Scheduled popup: see ops\register_daily_popup.ps1

$ErrorActionPreference = "Stop"
try { chcp 65001 > $null } catch {}
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
Set-Location -Path $PSScriptRoot

Write-Host ("=" * 68) -ForegroundColor DarkCyan
Write-Host " ESIS-QUANT  --  today's report" -ForegroundColor Cyan
Write-Host ("=" * 68) -ForegroundColor DarkCyan

# Health first: a stale suggestion from a dead job looks identical to a fresh
# one. If the pipeline is broken, say so loudly before showing any numbers.
$health = (py -3.11 -m ops.health) -join "`n"
if ($health -match "PIPELINE PROBLEM") {
    Write-Host $health -ForegroundColor Red
    Write-Host ""
} else {
    Write-Host $health -ForegroundColor DarkGray
}

# The report is read-only and also saves today's snapshot for the learning log.
py -3.11 -m ops.daily_report

# Generate today's order ticket (just a file -- no DB change, no trade).
Write-Host ""
Write-Host "Writing today's order ticket (what to buy + how much)..." -ForegroundColor Cyan
py -3.11 -m ops.order_ticket write

# Safety: never act unless a real person is here to answer. If this script is
# ever run non-interactively (no console, or piped/empty input), it must do
# NOTHING -- it must never place trades on its own.
if ([Console]::IsInputRedirected -or -not [Environment]::UserInteractive) {
    Write-Host "Non-interactive session -- report + ticket written, recording nothing." -ForegroundColor DarkGray
    exit 0
}

# Open the ticket so you can see what to do and fill in your real trades.
$asof = (py -3.11 -c "from ops.trade_list import latest_trade_list as L; print(L()['as_of_date'].iloc[0].date())").Trim()
$ticket = Join-Path $PSScriptRoot ("orders\orders_{0}.csv" -f $asof)
if (Test-Path $ticket) { try { Invoke-Item $ticket } catch {} }

# --- RECONCILIATION PROMPT ------------------------------------------------
# The book is the system's own arithmetic; the broker is ground truth. Every
# source of drift (a mistyped fill, a partial fill, a fee the model doesn't
# charge, a dividend) is individually silent, so an unchecked book stops
# describing reality without ever raising an error -- and every later
# suggestion is computed FROM that book. Reconciliation was built but nothing
# forced it, which made skipping it an oversight rather than a decision.
# Asking here makes it a decision.
$reconStale = (py -3.11 -c "from ops.reconcile import is_stale; print('STALE' if is_stale() else 'FRESH')").Trim()
if ($reconStale -eq "STALE") {
    Write-Host ""
    Write-Host "  BOOK NOT RECONCILED against Thndr recently." -ForegroundColor Yellow
    Write-Host "  Open Thndr and read the two numbers off the main screen." -ForegroundColor Yellow
    $sv = Read-Host "  Stocks value (EGP), or Enter to skip"
    if ($sv) {
        $cb = Read-Host "  Cash / balance (EGP)"
        if ($cb) {
            py -3.11 -m ops.reconcile totals $sv $cb
        } else {
            Write-Host "  Skipped -- both numbers are needed." -ForegroundColor DarkGray
        }
    } else {
        Write-Host "  Skipped. The book stays unverified until you run:" -ForegroundColor DarkGray
        Write-Host "     py -3.11 -m ops.reconcile totals <stocks> <cash>" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "What did you do with today's suggestion?" -ForegroundColor Yellow
Write-Host "  [1] ACCEPT  - take the system's book (records a rebalance to the target)"
Write-Host "  [2] DECLINE - skip this suggestion (nothing changes)"
Write-Host "  [3] OWN     - you traded your own view"
Write-Host "  [4] LATER   - decide later"
Write-Host "  [5] RECORD A REAL FILL - enter one actual trade you placed (buy/sell)"
Write-Host "  [6] APPLY FILLED TICKET - save all the trades you wrote in the ticket file"
Write-Host "  [7] RECONCILE - check the book against Thndr (stocks value + cash)"
$choice = Read-Host "Enter 1/2/3/4/5/6/7"

switch ($choice) {
  "1" {
        # ACCEPT rebalances your tracked book -- require an explicit typed YES so
        # it can never fire by accident or from a stray keystroke.
        Write-Host "This records a full rebalance to the system's target book." -ForegroundColor Yellow
        $confirm = Read-Host "Type YES (capitals) to confirm, anything else to cancel"
        if ($confirm -ceq "YES") { py -3.11 -m ops.portfolio_tracker accept }
        else { Write-Host "Cancelled -- nothing recorded." -ForegroundColor DarkGray }
      }
  "2" { py -3.11 -m ops.portfolio_tracker decline }
  "3" {
        $note = Read-Host "One-line note about your own trade"
        py -3.11 -m ops.portfolio_tracker own "$note"
      }
  "5" {
        # record the fill you actually got at your broker, so tracking is real
        $side   = Read-Host "buy or sell"
        $ticker = Read-Host "ticker (e.g. OLFI)"
        $shares = Read-Host "shares"
        $price  = Read-Host "fill price (EGP)"
        py -3.11 -m ops.portfolio_tracker $side $ticker $shares $price
      }
  "6" {
        Write-Host "Make sure you SAVED the ticket file after filling PAID_EGP + FEE_EGP." -ForegroundColor Yellow
        Write-Host "  PAID_EGP = total paid for the shares (exclude fee)   FEE_EGP = Thndr commission" -ForegroundColor DarkGray
        Read-Host "Press Enter when saved"
        py -3.11 -m ops.order_ticket apply
        # Fills just changed the book, so the broker check is immediately due.
        Write-Host ""
        Write-Host "Fills recorded -- reconcile now against Thndr." -ForegroundColor Yellow
        $sv2 = Read-Host "  Stocks value (EGP), or Enter to skip"
        if ($sv2) {
            $cb2 = Read-Host "  Cash / balance (EGP)"
            if ($cb2) { py -3.11 -m ops.reconcile totals $sv2 $cb2 }
        }
      }
  "7" {
        $sv3 = Read-Host "  Stocks value (EGP)"
        $cb3 = Read-Host "  Cash / balance (EGP)"
        if ($sv3 -and $cb3) { py -3.11 -m ops.reconcile totals $sv3 $cb3 }
        else { Write-Host "  Both numbers are needed." -ForegroundColor DarkGray }
      }
  default { Write-Host "No decision recorded. Run daily.ps1 again anytime." -ForegroundColor DarkGray }
}

Write-Host ""
Write-Host "Reminder: paper/real is YOUR call at the broker. The system never trades." -ForegroundColor DarkGray
Read-Host "Press Enter to close"
