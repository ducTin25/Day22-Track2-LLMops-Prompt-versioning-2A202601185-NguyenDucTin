# Evidence Guide — Day 22 Lab

This directory contains the artifacts required by `rubric.md`.

| File | What it demonstrates |
|---|---|
| `01_langsmith_traces.png` | LangSmith project showing at least 50 Step 1 traces |
| `02_prompt_hub.png` | Both prompt versions published in Prompt Hub |
| `02_ab_routing_log.txt` | 50 deterministic A/B-routed requests with `prompt-v1`/`prompt-v2` labels |
| `03_ragas_scores.png` | Terminal comparison of the four RAGAS metrics |
| `03_ragas_report.json` | Scores for both prompt versions and the target result |
| `04_pii_demo_log.txt` | Six PII validator cases, including clean and multi-PII inputs |
| `04_json_demo_log.txt` | Five JSON cases, including three automatic repairs and fallback |

## Result analysis

V1 is intentionally concise and grounded in the retrieved context. V2 adds a
structured expert response with an explicit evidence/uncertainty instruction.
Because both versions use the same retriever and context, faithfulness should
be comparable; V2 may improve answer clarity, while V1 may avoid unsupported
extra wording. The definitive comparison is recorded in
`../data/ragas_report.json` after the evaluation is run.

## How to generate the artifacts

Run the scripts from `src/` with the keys in the local `.env` file configured:

```powershell
python 01_langsmith_rag_pipeline.py
python 02_prompt_hub_ab_routing.py | Tee-Object ..\evidence\02_ab_routing_log.txt
python 03_ragas_evaluation.py
Copy-Item ..\data\ragas_report.json ..\evidence\03_ragas_report.json -Force
python 04_guardrails_validator.py | Tee-Object ..\evidence\04_guardrails_full_log.txt
```

Save the relevant LangSmith and terminal screenshots using the filenames in
the table above. Never commit `.env` or any API key.
