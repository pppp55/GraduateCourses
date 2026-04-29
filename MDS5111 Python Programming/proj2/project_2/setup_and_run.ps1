$ErrorActionPreference = 'Stop'

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.10 setup_and_run.py
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python setup_and_run.py
} else {
    Write-Error "Python not found. Please install Python 3.10.x (recommended: 3.10.20)."
}
