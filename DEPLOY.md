# Deploy Compliance Copilot to Render

About 5 minutes. You need your Groq key (https://console.groq.com/keys) and your Pinecone key (Pinecone console, **API Keys**).
No ingestion is needed: Pinecone already holds the indexed documents.

1. **Click the button.** Open the README on GitHub and click **Deploy to Render**, or open
   https://render.com/deploy?repo=https://github.com/rushimakwana112-cpu/compliance-copilot-india
2. **Sign in with GitHub** when Render asks, and allow it to read the repository.
3. **Paste the two keys.** Render shows the blueprint `compliance-copilot` and asks for:
   - `GROQ_API_KEY`: your Groq key
   - `PINECONE_API_KEY`: your Pinecone key

   Everything else (model `qwen/qwen3.8-27b`, index `compliance-copilot-cloud`, `PUBLIC_MODE=true`, threshold 0.38) is already filled in from `render.yaml`.
4. **Click Apply.** Wait until the log says `Your service is live` (the first build takes a few minutes). Copy the URL at the top, e.g. `https://compliance-copilot.onrender.com`.
5. **Check health.** Open `<url>/api/health` in the browser. Expected: `"status":"ok"` and `"public_mode":true` (also `"provider":"groq"`, `"pinecone":true`, `"docs":7`).
   If it says `"degraded"`, the `issues` list names the problem (usually a key pasted with a space).
   The first visit after 15 idle minutes can take up to a minute while the server wakes up.
6. **Ask the 3 test questions** on `<url>`:
   - *Who is eligible for gratuity under the Code on Social Security?*
   - *What is the definition of wages under the Code on Wages?*
   - *Is overtime paid at a higher rate than ordinary wages?*

   Each answer should cite the Code with a section and page. The upload and link boxes should not be visible.
7. **Paste the live URL back to Claude** in the project thread, so the README's `LIVE_URL_HERE` gets replaced.
