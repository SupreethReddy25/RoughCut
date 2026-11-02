# Regenerates everything under paper/data and paper/figures from the current code. Each step is logged and a failure does
# not stop the others.   powershell -ExecutionPolicy Bypass -File paper\scripts\run_all.ps1
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$env:PYTHONPATH = "backend;paper\scripts"
$env:ROUGHCUT_DATA_DIR = "C:\tmp\rc_paper"
$py = "backend\.venv\Scripts\python.exe"
$log = "paper\data\run_all.log"
"started $(Get-Date)" | Out-File $log -Encoding utf8
Remove-Item paper\data\exploration.json -ErrorAction SilentlyContinue  # it resumes from this file: start clean
$steps = @(
    "paper\scripts\rerun_e.py",
    "paper\scripts\run_experiments.py",
    "paper\scripts\run_exploration.py",
    "paper\scripts\paired_stats.py",
    "paper\scripts\calibration.py",
    "paper\scripts\learned_weights.py",
    "paper\scripts\new_features.py",
    "paper\scripts\measure_runtime.py",
    "paper\scripts\real_clips.py",
    "paper\scripts\make_figures.py",
    "paper\scripts\deep_baselines.py",
    "paper\scripts\deep_rhythm_tune.py",
    "paper\scripts\make_new_figures.py",
    "paper\scripts\make_diagrams.py"
)
foreach ($s in $steps) {
    "BEGIN $s $(Get-Date -Format HH:mm:ss)" | Out-File $log -Append -Encoding utf8
    $name = [IO.Path]::GetFileNameWithoutExtension($s)
    $p = Start-Process -PassThru -Wait -NoNewWindow -FilePath $py -ArgumentList $s -RedirectStandardOutput "paper\data\$name.out" -RedirectStandardError "paper\data\$name.err"
    "END $s exit=$($p.ExitCode) $(Get-Date -Format HH:mm:ss)" | Out-File $log -Append -Encoding utf8
}
"finished $(Get-Date)" | Out-File $log -Append -Encoding utf8
