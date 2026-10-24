param(
    [Parameter(Mandatory = $true)]
    [int]$PythonPid,
    [int]$IntervalSeconds = 30
)

$ErrorActionPreference = 'Stop'
$runLog = Join-Path $PSScriptRoot 'results_all_blocks\run.log'
$watchLog = Join-Path $PSScriptRoot 'results_all_blocks\watcher.log'

Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class PowerStateWatcher {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$lastRunLogLine = ''
$nextHeartbeat = (Get-Date).AddMinutes(5)
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') watcher started for Python PID $PythonPid" | Add-Content -LiteralPath $watchLog
try {
    while ($true) {
        $pythonProcess = Get-Process -Id $PythonPid -ErrorAction SilentlyContinue
        if (-not $pythonProcess) {
            "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Python PID $PythonPid exited" | Add-Content -LiteralPath $watchLog
            break
        }

        [PowerStateWatcher]::SetThreadExecutionState([uint32]2147483649) | Out-Null
        $currentRunLogLine = Get-Content -LiteralPath $runLog -Tail 1 -ErrorAction SilentlyContinue
        if ($currentRunLogLine -and $currentRunLogLine -ne $lastRunLogLine) {
            "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') progress | $currentRunLogLine" | Add-Content -LiteralPath $watchLog
            $lastRunLogLine = $currentRunLogLine
        }

        if ((Get-Date) -ge $nextHeartbeat) {
            $pythonProcess = Get-Process -Id $PythonPid -ErrorAction SilentlyContinue
            if ($pythonProcess) {
                "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') heartbeat | PID=$PythonPid CPU=$([math]::Round($pythonProcess.CPU, 0))s" | Add-Content -LiteralPath $watchLog
            }
            $nextHeartbeat = (Get-Date).AddMinutes(5)
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
}
catch {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') watcher error | $($_.Exception.Message)" | Add-Content -LiteralPath $watchLog
}
finally {
    [PowerStateWatcher]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') watcher released system-awake request" | Add-Content -LiteralPath $watchLog
}
