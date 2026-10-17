param([int]$Seconds = 60, [string]$Out = "")
# Samples total CPU utilisation once per second, and the per-process CPU-time delta over the window.
$before = @{}
Get-Process | ForEach-Object { try { $before[$_.Id] = @($_.Name, $_.CPU) } catch {} }
$c = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples $Seconds
$vals = $c | ForEach-Object { $_.CounterSamples[0].CookedValue }
$mean = ($vals | Measure-Object -Average).Average
$max = ($vals | Measure-Object -Maximum).Maximum
$after = Get-Process
$rows = foreach ($p in $after) {
  try {
    $b = $before[$p.Id]
    if ($b -and $p.CPU -ne $null) { [pscustomobject]@{Name = $p.Name; Delta = $p.CPU - $b[1]} }
  } catch {}
}
$top = $rows | Group-Object Name | ForEach-Object { [pscustomobject]@{Name = $_.Name; CpuSec = ($_.Group | Measure-Object Delta -Sum).Sum} } | Sort-Object CpuSec -Descending | Select-Object -First 5
$line = "idle-check {0}: window={1}s mean={2:N1}% max={3:N1}% top5=[{4}]" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Seconds, $mean, $max, (($top | ForEach-Object { "{0} ({1:N1} cpu-s)" -f $_.Name, $_.CpuSec }) -join "; ")
Write-Output $line
if ($Out) { Add-Content -Path $Out -Value $line -Encoding utf8 }
Write-Output ("MEAN={0:N2}" -f $mean)
