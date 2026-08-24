# Evidence Guide — Day 22 Lab

This directory contains the artifacts required by `rubric.md`.

| File | What it demonstrates |
|---|---|
| `01_langsmith_traces.png` | LangSmith project `track2-day22` showing 302 traces (at least 100 required overall) |
| `02_prompt_hub.png` | Both prompt versions published in Prompt Hub |
| `02_ab_routing_log.txt` | 50 deterministic A/B-routed requests with `prompt-v1`/`prompt-v2` labels |
| `03_ragas_scores.png` | Terminal comparison of the four RAGAS metrics |
| `03_ragas_report.json` | Scores for both prompt versions and the target result |
| `04_pii_demo_log.txt` | Six PII validator cases, including clean and multi-PII inputs |
| `04_json_demo_log.txt` | Five JSON cases, including three automatic repairs and fallback |

## Result analysis

V1 is intentionally concise and grounded in the retrieved context. V2 adds a
structured expert response with explicit evidence and uncertainty sections.
On the measured 50 QA pairs per version, V1 scored higher for faithfulness
(0.9425 vs 0.8407) and answer relevancy (0.9092 vs 0.8298). Both versions tied
for context recall (1.0000) and context precision (0.9417) because they use the
same retriever and retrieved contexts. The longer V2 structure likely creates
more opportunities for wording that is not directly supported by the context;
therefore V1 is the recommended prompt for this dataset.

## How to generate the artifacts

Run the scripts from `src/` with the keys in the local `.env` file configured:

```powershell
python 01_langsmith_rag_pipeline.py
python 02_prompt_hub_ab_routing.py
python 03_ragas_evaluation.py
python 04_guardrails_validator.py
```

Step 2 writes the A/B routing log automatically. Step 3 copies
`data/ragas_report.json` to `evidence/03_ragas_report.json`. Run
`python scripts/render_ragas_evidence.py` from the repository root to render
the measured report as `03_ragas_scores.png`. Step 4 writes the PII and JSON
evidence logs separately, so all generated artifacts match the latest run.

The two LangSmith screenshots must be captured from the real project and
Prompt Hub UI. Never commit `.env` or any API key.
