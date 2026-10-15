# Evaluation report

Not run yet. The eval needs Ollama and an indexed Pinecone index, so it runs on your machine:

```powershell
python -m eval.run --sweep 0.45,0.5,0.55,0.6,0.65
```

This file is overwritten with the metrics, a threshold sweep and a per-question table.
