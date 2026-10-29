# Verification Record

**Project:** SourceGrounded, AAI 6665 Module 7.3  
**Checked:** October 28, 2026  
**Environment:** Python 3.12 (implementation supports Python 3.10+)

## Checks performed on the packaged source

| Check | Observed result |
| --- | --- |
| Python unittest discovery | **26 tests passed; 0 failures** |
| Python compilation / syntax check | Passed |
| CLI offline retrieval | Passed; API-secret query returned S5 (cosine score 0.617213) |
| CLI missing API-key behavior | Passed; descriptive error without a network call |
| 36-case fixture balance | 12 answerable + 12 partial + 12 unanswerable |
| TF-IDF source-level recall@3 across 24 gold-bearing questions | **23/24 = 0.9583** |
| Unanswerable queries with nonempty retrieval | **3/12** |
| Live Groq GPT-OSS 120B answer generation | **Not run in build environment** — no configured provider API key |
| Human review of generated factual claims | **Not run** — requires real outputs and reviewers |
| Optional semantic retriever | **Not tested** — extra dependency and model download |
| Optional Streamlit visual UI | Syntax-checked; **not launched** in the build environment |

The only question with missed gold evidence under the stated query-recall definition was P03. False nonempty retrieval on unanswerable questions is a useful reminder that lexical similarity is not reliable proof of answerability.

## Settings

- Corpus: data/demo_sources.json, eight short original synthetic source passages.
- Benchmark: data/eval_36.json, 36 original synthetic questions.
- Retriever: TF-IDF cosine similarity with smoothed IDF, top K = 3, minimum cosine score = 0.08.
- Hosted generation if enabled: model openai/gpt-oss-120b, temperature 0.1, maximum completion tokens 1,200, low reasoning effort.
- API transport: Groq official Python SDK replaces the original urllib implementation. The original client was blocked with HTTP 403/1010 during user testing; the updated SDK pathway was verified with mock responses, not yet against a live account.
- Test model: controlled fake client, which deliberately returns known test responses. Fake-client outputs are **not** presented as evidence of LLM quality.

## Interpreting the results

Recall@3 is measured only for source retrieval, not generation. It tells us whether a document ID known to contain at least part of the answer appeared in the search results. It cannot tell whether a generated claim was truthful. For example, a valid source ID can be attached to a fabricated claim. The guarded implementation detects unknown source IDs and malformed outputs but still requires human fact checking.

The proposal's central hypothesis — that guarded RAG improves factual support and appropriate abstention compared to other conditions — has **not been tested** by these offline results. Run the comparison and label its outputs to make such a claim.

## Reproduction

~~~bash
python -m unittest discover -s tests -v
python -m sourcegrounded.cli retrieve "Where should API secrets be stored?"
python -m sourcegrounded.cli evaluate-retrieval
~~~

For online evaluation, configure the provider first and follow README.md and EVALUATION_GUIDE.md. Do not invent scores for experiments that have not been run.
