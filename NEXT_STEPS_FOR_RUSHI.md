# Next steps for Rushi

Only three things need you. Everything else (tests, re-index, eval, README numbers, merge) is done or scripted. The long version, with the demo-video plan, is in [docs/manual-steps.md](docs/manual-steps.md).

## 1. Run the script

In PowerShell, in the project folder:
```powershell
.\scripts\run_all.ps1
```
Expected: a green `OK:` line after each step and `All done.` at the end. It switches to the deployed (default) branch, pulls, installs, runs the tests (`243 passed`), re-indexes Pinecone, runs the eval, writes the numbers into the README table, commits and pushes.
If it stops, the red `STOPPED at:` line names the step and what to check. If Windows says scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

## 2. Check the null and flagged rates in `data/rates.yaml`

No official PDF was reachable from the build machine, so nothing was filled from a document. Please fill these from the official text (value, `effective_from`, page in `section`, `url`), then run the script again:

| Field | Status | Take it from |
|---|---|---|
| `pf.wage_ceiling` 25000 (current) | **check**: `effective_from` and page are null | Gazette notification of 17 Sept 2026 |
| `pf.wage_ceiling` 15000 (historical) | **check**: `effective_to` and the para are null; source marked "check" | EPF Scheme, 1952 |
| `pf.employee_rate`, `pf.employer_rate` | null | EPF Scheme, 2026 |
| `pf.eps_rate`, `pf.eps_wage_cap` | null | EPS, 2026 |
| `pf.edli_rate` | null | EDLI Scheme, 2026 |
| `esic.wage_ceiling`, `esic.employee_rate`, `esic.employer_rate` | null | ESIC notifications |
| `gratuity.days_per_year`, `wage_days_divisor`, `min_years_permanent`, `min_years_fixed_term`, `round_up_after_months`, `max_amount` | null | Code on Social Security, 2020 (gratuity section); gratuity ceiling notification |
| `professional_tax.gujarat` and `.maharashtra` slabs | null | each state's Professional Tax schedule |

To index those documents too, paste their links into the commented rows at the bottom of `data/sources.csv` (remove the `# `); the script picks them up. Check the result at http://localhost:8000/api/rates after `uvicorn app.main:app --reload`: filled fields show their value and source, the rest `null`.

## 3. Deploy: click the button and paste 4 keys

Click **Deploy to Render** at the top of the README (sign in with GitHub). Paste:
- `GROQ_API_KEY`: from https://console.groq.com/keys
- `GROQ_MODEL`: `qwen/qwen3.8-27b`
- `PINECONE_API_KEY`: from the Pinecone console, **API Keys**
- `PINECONE_INDEX`: `compliance-copilot-cloud`

Click **Apply**, wait for `Your service is live`, then check (use your URL):
```powershell
Invoke-RestMethod https://compliance-copilot.onrender.com/api/health
```
Expected: `status : ok`, `provider : groq`, `public_mode : True`. The first call after 15 idle minutes can take up to a minute.
