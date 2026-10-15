# Next steps for Rushi

Only three things need you. Everything else (tests, re-index, eval, README numbers, merge) is done or scripted. The long version, with the demo-video plan, is in [docs/manual-steps.md](docs/manual-steps.md).

## 1. Run the script

In PowerShell, in the project folder:
```powershell
.\scripts\run_all.ps1
```
Expected: a green `OK:` line after each step and `All done.` at the end. It switches to the deployed default branch (`main`), pulls, installs, runs the tests (`368 passed`), re-indexes Pinecone, runs the eval, writes the numbers into the README table, commits and pushes.
If it stops, the red `STOPPED at:` line names the step and what to check. If Windows says scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

## 2. Confirm the rates marked "check" in `data/rates.yaml`

The calculators and due dates now work with figures taken from official documents (each one is cited in the answer). The ones below were read from a copy of the Gazette text because the official site blocked the build machine. Open the official PDF, confirm the figure and the para, then delete ` - check` from its `source_doc` and add the `url`:

| Field | Value used | Confirm in |
|---|---|---|
| `pf.employee_rate`, `pf.employer_rate` | 0.12 each | EPF Scheme, 2026 (G.S.R. 525(E)) para 18(2), and S.O. 3582(E) of 1 July 2026 (some establishments: 10%) |
| `pf.eps_rate` | 0.0833 | S.O. 3580(E) of 1 July 2026 (egazette ID 274111) |
| `pf.eps_wage_cap` | 25000 from 17 Sept 2026 | EPS, 2026 (G.S.R. 527(E)) paras 4(1), 11(3) |
| `pf.edli_rate` | 0.005 | S.O. 3581(E) of 1 July 2026 (egazette ID 274104) |
| `pf.wage_ceiling` 15000 (historical) | until 16 Sept 2026 | the earlier ceiling (S.O. 2702(E) of 29 May 2026 may have changed it) |
| `gratuity.wage_days_divisor`, `round_up_after_months` | 26, 6 | Code on Social Security, 2020 s.53 (Explanation and s.53(2)) |
| `due_dates.pf` | 15th of next month | EPF Scheme, 2026 paras 20(1), 29(2) |

**New figures (income tax, bonus, overtime, leave), read on the official sites:** the `income_tax` slabs, standard deduction, rebate, cess, surcharge and deduction limits come from incometaxindia.gov.in pages for the Income-tax Act, 2025 and Rules, 2026; HRA uses rule 279, which now lists 8 cities at 50% (Mumbai, Kolkata, Delhi, Chennai, Hyderabad, Pune, Ahmedabad, Bengaluru). Bonus, overtime and leave come from the Code on Wages, the Wage Rules 2026 and the OSH Code on labour.gov.in. Spot-check them at `/api/rates`.

**Fill these when you find the official notification (answers say "not configured" until then):**

| Field | What it is | Where to look |
|---|---|---|
| `bonus.eligibility_wage_ceiling` | wage limit for bonus eligibility (secondary sites say ₹21,000, S.O. 4710(E), 25 Aug 2026) | egazette.gov.in / labour.gov.in |
| `bonus.calculation_ceiling` | wages used for bonus when actual wages are higher (secondary sites say ₹7,000 or the minimum wage) | same notification, under s.26(2) |
| `income_tax.gratuity_exemption_limit` | tax-free gratuity limit under the 2025 Act | incometaxindia.gov.in notifications |
| `income_tax.leave_encashment_exemption_limit` | tax-free leave encashment on retirement | same |
| Gujarat minimum wages | not in rates.yaml; the FAQ points to col.gujarat.gov.in | https://col.gujarat.gov.in/e-citizen-rate26-2.htm |

**Still empty (answers say "not configured"):** `gratuity.max_amount` (no ceiling notified under s.53(3) yet; the gratuity calculator shows the amount uncapped and says so), Maharashtra Professional Tax slabs and due date (separate slabs for women and a February rule; check mahagst.gov.in).

Check the result at http://localhost:8000/api/rates after `uvicorn app.main:app --reload`, or on the live site at `/api/rates`.

## 3. Deploy: click the button and paste 2 keys

Follow [DEPLOY.md](DEPLOY.md). Only `GROQ_API_KEY` and `PINECONE_API_KEY` are asked for; the model, index and public mode are set in `render.yaml`.
