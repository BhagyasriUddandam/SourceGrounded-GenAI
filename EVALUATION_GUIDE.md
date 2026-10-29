# Benchmark and Human-Review Guide

## Research question

When asked a question that is fully answerable, partially answerable or unanswerable using a controlled document collection, does retrieval plus structured citation checking reduce unsupported generated statements and encourage appropriate abstention?

### Experimental conditions

- **A (model-only):** no retrieved context. Expected weakness: may use general knowledge to answer fictional source-specific facts.
- **B (basic RAG):** retrieved passages in context. Expected weakness: irrelevant context and fabricated source citations may still be used.
- **C (guarded RAG):** retrieved passages plus structured constrained prompt, citation-ID membership checks, one clarification and abstention. Expected strength: clearer provenance and structural error detection; expected limitation: semantic misinformation can still slip through.

## Data

The eight synthetic passages S1–S8 exactly match the proposal's illustrative pilot wording. The 36 case IDs are A01–A12 (answerable), P01–P12 (partial) and U01–U12 (unanswerable). For partial cases, a good answer states the supported subfact, identifies what is unknown, and cites the relevant passage. For unanswerable cases, a good guarded response abstains without hallucinating source facts.

Gold labels are **source IDs** at document level; they are not automatic answer judgments. A case may have one or more applicable sources.

## Step 1: Offline retrieval

~~~bash
python -m sourcegrounded.cli evaluate-retrieval
~~~

For an answerable or partial question with gold set G and retrieved source set R:

- Query recall@3 indicator = 1 if the intersection of G and R is nonempty, otherwise 0.
- Overall query recall@3 = mean of the indicators across gold-bearing items.
- Unanswerable-with-nonempty-retrieval = count of items labeled unanswerable whose retrieval returned any passage.

This source-level query recall is not average passage recall when multiple gold passages are required. Report the exact definition.

## Step 2: Live generation

Set GROQ_API_KEY in a local .env file and verify your provider supports the chosen model. For a trial run:

~~~bash
python -m sourcegrounded.cli evaluate --limit 3 --repeats 1 --out outputs/pilot --confirm-api-calls
~~~

For the complete research plan:

~~~bash
python -m sourcegrounded.cli evaluate --limit 36 --repeats 2 --out outputs/full --confirm-api-calls
~~~

The runner logs each question, mode, repetition, generated text, citations, retrieval, status and provider usage. Up to 216 requests are possible and may incur costs. Withheld or early-abstaining answers are still meaningful outputs and should be reviewed.

## Step 3: Human annotation

Open human_annotation_template.csv in a spreadsheet and assess every answer against the corresponding original passage. Fill these columns:

| Column | Human coding rule |
| --- | --- |
| supported_claims | Number of distinct factual claims entailed by the provided source; 0 if none |
| unsupported_claims | Number of assertions contradicted by or unsupported in the provided evidence |
| valid_citations | Count of citations whose exact associated claim is supported by the cited passage |
| total_citations | Number of citation occurrences in the answer; 0 if none |
| usefulness_1_to_5 | 1 unusable, 2 poor, 3 partially useful, 4 useful, 5 complete and clear |
| abstention_appropriate_0_or_1 | 1 if abstaining/limited answer matches evidence availability; 0 if inappropriate |
| reviewer_notes | Evidence excerpts or rationale, especially for partial/unanswerable cases |

Do not equate a citation's ID existing in the corpus with a **valid citation**. The citation's associated claim must actually be supported. For model-only answers, there may be no document citations. Support should be assessed against the benchmark evidence or trusted primary material according to your annotation protocol, not merely the model's confidence.

For an answer containing no factual claims (e.g., an appropriate abstention), use 0 supported and 0 unsupported. Its claim support rate is undefined; do not turn a 0/0 denominator into 100%. You can still score appropriateness and usefulness.

If a row was not reviewed, **leave every human label blank**, not zero. At least two independent reviewers would strengthen inter-rater reliability; record disagreements. If only one reviewer is available, state that as a limitation.

## Step 4: Score

~~~bash
python -m sourcegrounded.cli score-human --csv outputs/full/human_annotation_template.csv
~~~

Aggregate definitions:

- Supported-claim rate = total supported claims / (total supported + total unsupported claims).
- Unsupported-claim rate = total unsupported / all reviewed factual claims.
- Citation precision = total claim-supported citations / all reviewed citation occurrences.
- Mean usefulness = mean entered 1–5 ratings.
- Appropriate abstention = mean entered binary abstention judgments.

Rows without valid human labels are excluded from the corresponding metric. Report the labeled row count and missingness alongside comparisons.

## Reliability and reporting

The small synthetic corpus does not represent all academic content, and eight short passages are easier to retrieve than a real research archive. The 36 questions were written for this project and have not been independently validated. Randomness can affect hosted generation, even at temperature 0.1. Comparison across conditions is valid only if the same corpus, questions, model and settings are retained. If the provider/model version changes, record that explicitly.

Report retrieved source IDs, actual claims and citations, reviewer evidence, useful examples of hallucination and abstention, and uncertainty. Distinguish observations from hypotheses and avoid claiming clinical, legal or academic policy authority.
