# Alignment with the Module 3 Midterm Proposal

This implementation is scoped to the submitted PDF titled *Source-Grounded Conversational Question Answering Using Retrieval-Augmented Generation*. The proposal emphasizes evaluating whether retrieved evidence and a source-checking policy improve text generation under incomplete evidence.

| Proposal element | Source-code implementation | Completion status |
| --- | --- | --- |
| Application: academic/technical educational QA | sourcegrounded/pipeline.py and optional Streamlit app | Implemented |
| Condition A: model-only | model-only generation prompt, no retrieval | Implemented; live output needs API key |
| Condition B: basic RAG | TF-IDF top-K passages + prompting | Implemented; live output needs API key |
| Condition C: guarded RAG | JSON output, source-ID verification, abstain, one clarification | Implemented as structural checks; semantic verification remains human |
| Hosted language model | Proposal selected Llama 3.3 70B; default updated to Groq GPT-OSS 120B after free/developer-tier Llama access was deprecated | Documented deviation required by model availability; compare all three modes with the same newer model |
| Document chunking and provenance | corpus.py, SHA-256, manifest, source metadata | Implemented |
| Lexical retrieval pilot | Pure Python TF-IDF with cosine similarity | Implemented and tested |
| Optional semantic embeddings | Sentence-Transformers retriever | Implemented as an optional dependency; not benchmarked in this package |
| Eight pilot source snippets | data/demo_sources.json, S1 through S8 | Implemented with proposal wording |
| Real public primary literature | data/technical_notes.json contains original paraphrases and paper links | Partial: links/summaries, not original full-text ingested |
| Benchmark: 12 answerable, 12 partial, 12 unanswerable | data/eval_36.json | Implemented |
| Three modes × 36 questions × two repeats | evaluate command with limit 36 and repeats 2 | Ready; actual live execution needs API key |
| Retrieval recall@3 | evaluate-retrieval CLI diagnostic | Implemented and measured offline |
| Claim support, unsupported claims, citation precision, utility | Manual review CSV + score-human | Implemented, awaiting human annotations |
| Latency, model usage | Recorded from actual provider responses | Implemented, awaiting provider execution |
| Deployment instructions | README, pyproject.toml, .env.example | Complete |
| Tests and source-code documentation | 26 unittest cases + docs | Complete |

## Distinguish proposal hypotheses from observations

The project's research hypothesis is that guarded RAG will produce fewer unsupported claims and more appropriate abstentions than basic RAG and a model-only baseline. The hypothesis is **not yet established** by this source-code deliverable. The environment used to assemble the project did not contain a live model API key.

What has been observed is the offline TF-IDF search diagnostic and successful tests of code paths using fake provider responses. The offline benchmark is synthetic and only tests retrieval coverage and false nonempty hits. A final report of actual answer quality requires running the model and manually labeling the outputs.

## Suggested reproducibility settings

- Fixed 36-question JSON fixture; do not revise it after inspecting generated answers without documenting changes.
- Three modes use the same model and temperature 0.1.
- Retriever: TF-IDF, top K = 3, minimum similarity = 0.08.
- Two independent generations per question per mode when evaluating repeat variability.
- Retain original source text/hash and log timestamp, model version, token usage, and answer status.
- When labeling claim support, reviewers should see source text, not only source IDs. Mark unsupported assertions even when they cite a real passage.
- Summarize disagreements between independent reviewers and do not silently change labels to improve reported scores.

## Source authority

The eight pilot snippets are original synthetic examples from the proposal, not authentic policy documents. The separate research summary corpus includes links to peer-reviewed/public primary research and the NIST AI 600-1 document. For external validation, import actual authorized primary documents with the ingest command and a reviewed manifest.
