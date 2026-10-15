<#
  Compliance Copilot: everything to run on your PC, in one go.

    .\scripts\run_all.ps1

  Steps: activate .venv -> git pull -> pip install -> pytest -> re-index Pinecone -> eval
         -> copy the eval numbers into README.md -> commit -> push.
  It stops at the first step that fails and says which one.

  Options:
    -SkipIngest    don't re-index (documents unchanged since the last run)
    -NoPush        commit locally but don't push
    -Delay 8       seconds between eval questions (Groq free tier)

  If Windows says running scripts is disabled, run this once, then try again:
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
param(
    [switch]$SkipIngest,
    [switch]$NoPush,
    [double]$Delay = 8
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Fail([string]$step, [string]$hint) {
    Write-Host ""
    Write-Host "STOPPED at: $step" -ForegroundColor Red
    if ($hint) { Write-Host $hint -ForegroundColor Yellow }
    Write-Host "Fix the problem above, then run .\scripts\run_all.ps1 again." -ForegroundColor Yellow
    exit 1
}

# Runs one step. For programs (git, pip, python) the exit code decides; for PowerShell commands, an error does.
function Step([string]$name, [scriptblock]$command, [string]$hint = "") {
    Write-Host ""
    Write-Host "==> $name" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    try {
        & $command
    } catch {
        Write-Host $_ -ForegroundColor Red
        Fail $name $hint
    }
    if ($LASTEXITCODE -ne 0) { Fail $name "$hint (exit code $LASTEXITCODE)" }
    Write-Host "OK: $name" -ForegroundColor Green
}

Step "Check the project setup" {
    if (-not (Test-Path ".venv\Scripts\Activate.ps1")) { throw "No .venv folder. Create it with: python -m venv .venv" }
    if (-not (Test-Path ".env")) { throw "No .env file. Create it with: Copy-Item .env.example .env; notepad .env" }
}

Step "Activate the virtual environment" { . .\.venv\Scripts\Activate.ps1 }

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
Step "Pull the latest code ($branch)" { git pull origin $branch } `
    "If git reports a conflict or local changes, commit or stash them (git stash) and run again."

Step "Install packages" { python -m pip install --quiet -r requirements-dev.txt }

Step "Run the tests" { python -m pytest -q } "A test failed: send the output above to Claude before going on."

if (-not $SkipIngest) {
    Step "Re-index the documents in Pinecone (takes a few minutes)" { python -m ingest.run --reindex } `
        "Check PINECONE_API_KEY and PINECONE_INDEX in .env, and that the URLs in data\sources.csv open in a browser."
}

Step "Run the evaluation (about 8 s per question)" { python -m eval.run --delay $Delay } `
    "Check GROQ_API_KEY and GROQ_MODEL in .env. If many questions were rate limited, run again with -Delay 15."

Step "Copy the eval numbers into README.md" { python -m scripts.update_readme_eval }

Step "Commit the results" {
    $files = @("README.md", "eval/report.md", "data/rates.yaml", "data/sources.csv")
    if (Test-Path "data/manifest.json") { $files += "data/manifest.json" }
    git add -- $files
    if ($LASTEXITCODE -ne 0) { return }
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { Write-Host "Nothing changed, nothing to commit."; return }
    git commit -m "Update eval results ($(Get-Date -Format 'yyyy-MM-dd'))"
}

if (-not $NoPush) {
    Step "Push to GitHub ($branch)" { git push origin $branch } "Check that you are signed in to GitHub (git push asks for it)."
}

Write-Host ""
Write-Host "All done. The eval report is in eval\report.md and the README table is updated." -ForegroundColor Green
