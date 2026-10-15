# Manual steps in detail

The short version is [NEXT_STEPS_FOR_RUSHI.md](../NEXT_STEPS_FOR_RUSHI.md); `scripts\run_all.ps1` does step 1 for you, and PR #1 is already merged.

Everything below runs in **PowerShell** in the project folder (`cd` to where you cloned `compliance-copilot-india`).

## 1. Pull, test, re-run the eval, update the README

```powershell
.\.venv\Scripts\Activate.ps1
git checkout feature/groq-cloud
git pull origin feature/groq-cloud
pip install -r requirements-dev.txt
python -m pytest
python -m eval.run
```
Expected: `python -m pytest` ends with `240 passed`. `python -m eval.run` prints a table titled *Compliance Copilot eval* and ends with `Report written to …\eval\report.md`; questions whose document isn't indexed show as *skipped (doc not ingested)*.

Copy the numbers from `eval\report.md` into the table between the `eval-results` markers in `README.md` (`scripts\run_all.ps1` does this for you), then:
```powershell
git add README.md eval\report.md
git commit -m "Update eval results"
git push origin feature/groq-cloud
```

## 2. Fill `data/rates.yaml` from the official documents

Every figure is a list of dated versions. Fill `value`, `effective_from`, `section` (page or para) and `url`; leave `null` anything the document doesn't state.

| Field | Fill from |
|---|---|
| `pf.wage_ceiling` (25000, current) | `effective_from` and page: Gazette notification of 17 Sept 2026 |
| `pf.wage_ceiling` (15000, historical) | `effective_to` (the day before the new ceiling) and `source_doc`/para: EPF Scheme, 1952 |
| `pf.employee_rate`, `pf.employer_rate` | EPF Scheme, 2026 (notified 1 July 2026) |
| `pf.eps_rate`, `pf.eps_wage_cap` | EPS, 2026 |
| `pf.edli_rate` | EDLI Scheme, 2026 |
| `esic.wage_ceiling`, `esic.employee_rate`, `esic.employer_rate` | ESIC wage-ceiling and contribution-rate notifications |
| `gratuity.*` (days, divisor, minimum years, rounding, max amount) | Code on Social Security, 2020, gratuity section, and the gratuity ceiling notification |
| `professional_tax.gujarat.slabs` | Gujarat Professional Tax schedule |

Check a value after filling it:
```powershell
uvicorn app.main:app --reload
```
Then open http://localhost:8000/api/rates . Expected: each field you filled shows its `value` and `source_doc`; unfilled ones show `"value": null`.

## 3. Add the missing documents and ingest them

In `data/sources.csv`, at the bottom, each commented row needs its official PDF link (replace `PASTE_URL`) and effective date, then delete the leading `# `:
EPF Scheme 2026 (and EPS / EDLI 2026), the ESIC wage-ceiling notification, the Gujarat Professional Tax schedule, the CBDT circular on TDS on salaries.

```powershell
python -m ingest.run --dry-run
python -m ingest.run
```
Expected: the dry run lists the new documents as *would index*; the real run prints one line per document with its chunk count. Re-run `python -m eval.run`: the questions that were *skipped (doc not ingested)* are now asked.

## 4. Merge PR #1 on GitHub

Open https://github.com/rushimakwana112-cpu/compliance-copilot-india/pull/1 , check that the checks are green, click **Merge pull request**, then **Confirm merge**.

## 5. Deploy on Render

Follow [DEPLOY.md](../DEPLOY.md): click the button, sign in with GitHub, paste `GROQ_API_KEY` and `PINECONE_API_KEY`, click **Apply**, then check `/api/health`.

## 6. Record a 2-minute demo video

Use Windows **Snipping Tool → Record** (or Xbox Game Bar, `Win + Alt + R`) on the deployed site:
1. (0:00–0:15) The page and its promise: Indian payroll law, answered with proof.
2. (0:15–0:45) Ask *"Who is covered for gratuity under the SS Code?"* and click a citation to show the source passage.
3. (0:45–1:10) Ask *"Calculate PF for basic ₹30,000"*: the answer names the wage ceiling and notification it used.
4. (1:10–1:30) Ask *"Gratuity rule in Dubai?"*: it refuses instead of guessing.
5. (1:30–2:00) Ask *"Which sample employees are above the ESIC limit?"* and show the SQL, then mention the eval numbers in the README.
