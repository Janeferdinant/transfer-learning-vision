param([string]$Log, [int]$MaxWaitMin = 30)
# Idle gate: 60 s CPU sample; if mean > 15 % wait 5 min and re-check, for up to MaxWaitMin minutes.
$start = Get-Date
$tool = Join-Path $PSScriptRoot 'idle_check.ps1'
$n = 0
while ($true) {
  $n++
  $out = & $tool -Seconds 60 -Out $Log
  $line = ($out | Where-Object { $_ -like 'idle-check*' }) -join ''
  $mean = [double](($out | Where-Object { $_ -like 'MEAN=*' }) -replace 'MEAN=', '')
  Write-Output $line
  if ($mean -le 15) { Write-Output "GATE: PASS (check $n, mean $mean)"; Add-Content $Log "GATE: PASS (check $n, mean $mean)"; break }
  if (((Get-Date) - $start).TotalMinutes -ge $MaxWaitMin) {
    Write-Output "GATE: FAIL after $n checks; running anyway with background load recorded (last mean $mean)"
    Add-Content $Log "GATE: FAIL after $n checks over $MaxWaitMin min; running anyway, last mean $mean"
    break
  }
  Add-Content $Log "waiting 300 s before re-check"
  Start-Sleep -Seconds 300
}
