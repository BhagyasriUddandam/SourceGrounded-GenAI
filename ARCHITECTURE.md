# Software Architecture and Engineering Decisions

## Objective

Build an understandable, reproducible, text-generation question-answering prototype with defensible source provenance and a controlled comparison of three conditions: model-only, basic retrieval-augmented generation (RAG), and guarded RAG.

## Data flow

~~~text
               Approved local TXT / MD / HTML / PDF documents
                                 |
                          Corpus + manifest
                                 |
                  Validate IDs, hash source text
                                 |
                    Word chunking with overlap
                                 |
                 TF-IDF index [optional: embeddings]
                                 |
User question ----------------> Top-K ranked passages
     |                              |
     |                         Passage text and IDs
     |                              |
     +--------> Model-only       Basic RAG        Guarded RAG
                     |              |                 |
              Generic prompt  Context prompt    Grounded JSON prompt
                     |              |                 |
                     +-------- Hosted chat model ------+
                                    |
               Raw answer / citations / provider usage
                                    |
                   Guarded: parse structured JSON,
                   validate retrieved citation IDs,
                   enforce one clarification maximum
                                    |
                 UI/CLI JSON; batch logs; human review
~~~

The model-only condition intentionally bypasses retrieval when answering. For basic RAG, all generated text is shown; a warning marks unknown cited IDs but does not suppress the answer. For guarded RAG, invalid JSON, nonretrieved source IDs or missing inline citation markers cause an explicit withheld-answer response. When no passages pass the score threshold, guarded RAG abstains without calling the model.

## Modules

### Corpus module

Local ingestion is controlled by an optional manifest keyed by filename. Source records include a unique stable ID, title, text, source URL, author, year, source type, retrieved date and SHA-256 content digest. Import checks invalid IDs, duplicate IDs, missing content and mismatched hashes. The source digest supports audit/reproducibility, not authenticity or copyright clearance. Each chunk retains a source ID and digest.

### Retrieval

Default retrieval is TF-IDF cosine similarity, coded directly in Python using a word tokenizer, smoothed IDF and vector normalization. A minimum threshold suppresses unrelated zero-score hits. A small top K avoids sending excess text. This is lightweight and reproduces the lexical proof-of-concept from the proposal. It is not a state-of-the-art retriever.

The optional semantic retrieval implementation uses Sentence-Transformers and normalized sentence vectors. It is intentionally opt-in because it adds substantial dependencies, CPU cost and a first-use model download. Both retrievers return the same Hit interface, allowing controlled comparisons.

### Hosted generation adapter

The chat client now uses Groq's **official Python SDK** to access its chat-completions interface over HTTPS, taking the secret from GROQ_API_KEY. This replaces the first prototype's direct urllib client, which encountered Cloudflare HTTP 403/1010 client-signature blocks. SDK retries handle rate limits and transient provider errors. Temperature is fixed at 0.1, with a 1,200-token completion limit and low reasoning effort for GPT-OSS models. Provider usage and response time are captured where supported.

Live prompts and retrieved context leave the local machine and reach the provider. Only publicly licensed or explicitly permitted local sources should be used. Local development may use an HTTP localhost endpoint; nonlocal insecure HTTP is rejected.

### Prompting and guardrails

Three system prompts deliberately keep the same application domain but vary access to evidence and postprocessing. Guarded mode requests JSON with answer, cited_source_ids and evidence_status. The status may be supported, partial, insufficient or clarify.

The one-clarification budget is tracked in ConversationState and resets for each independent evaluation question. The developer-facing UI preserves it across a chat session. Retrieved documents are labeled untrusted data, not executable instructions. This mitigates some prompt injection patterns but cannot guarantee protection against all malicious source content.

**Guardrail boundary:** Source-ID membership, JSON schema and inline citation checks are **structural**, not semantic. An answer might still contain an unsupported statement next to a valid source ID. The methodology therefore requires independent human verification of actual claims against source text.

### Evaluation

The benchmark has 12 answerable, 12 partially answerable and 12 unanswerable synthetic questions. Each has manually chosen gold source IDs, not a generated answer key. Retrieval-only evaluation measures query-level any-gold recall@K for 24 cases. It additionally counts nonempty retrieval on unanswerable cases. This explicitly distinguishes a search miss from a generation hallucination.

Online evaluation iterates through all three modes and a configurable number of repetitions. It records text, source IDs, status, latency and any usage metadata. A second independent human-labeling step measures supported and unsupported claims, citation precision, usefulness, and appropriate abstention. This ensures implementation artifacts cannot masquerade as evaluation evidence.

## Scope and tradeoffs

| Decision | Benefit | Limitation |
| --- | --- | --- |
| Pure Python TF-IDF | No API or external dependency for index | Poor paraphrase handling |
| Hosted GPT-OSS 120B API | No local GPU requirement | External cost, privacy, nondeterminism; model differs from initial Llama 3.3 proposal due to provider deprecation |
| 8 synthetic pilot sources | Reproducible, auditable known facts | Not evidence of real-world generalization |
| 36 human-authored cases | Balanced answerability test | Small and not independently benchmarked |
| Citation-ID validator | Detects hallucinated identifiers | Cannot establish actual claim support |
| Human review template | Honest factuality/utility scoring | Requires time and multiple reviewers for reliability |
| CLI + optional UI | Deployable without proprietary platform | UI has no authentication or production controls |

## Risks and mitigation

1. **Confabulation:** Withhold invalid guarded outputs; include abstention; manually audit factual support.
2. **Source authority:** Record provenance and point to primary papers; synthetic demonstration documents are clearly labeled.
3. **Privacy:** Process approved local documents, keep credentials in environment variables, send only allowed questions/passages to provider.
4. **Evaluation leakage:** Separate test questions from model-generated responses; avoid adding model responses to the source index.
5. **Prompt injection:** System instruction treats passage text as untrusted; structural validation helps but does not completely prevent attacks.
6. **Provider dependency:** Offline retrieval and tests work without credentials; descriptive errors explain live API requirements.
7. **Research ethics:** Do not claim unrun experiments, fabricate citations or interpret source-ID membership as validated truth.

## Future extensions

Add a calibrated answerability classifier; true entailment/claim verification with separate human review; multilingual evaluation; persistent FAISS or vector-store index; authentic licensed document collection; stratified inter-rater agreement; provider-cost dashboards; proper authentication and monitoring if deployed publicly. None of these production features are claimed as implemented.
