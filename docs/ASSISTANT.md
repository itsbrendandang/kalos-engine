# The assistant

The assistant helps you find your way around a campaign. It has three parts:

- **Next step.** One thing to do next for a campaign, and why. This part uses no model. It reads the campaign record and nothing else.
- **Ask.** A question box. The answer is built only from your own data and from these docs, and each sentence points to where it came from.
- **SOP drafts.** A starting draft for a procedure, filled in from the campaign record, with blanks for everything the lab has to decide.

The code is in `kalos/assist/`.

## What it can answer

It can answer questions about things Kalos has recorded for your tenant:

- your run sheets: the target, the inputs, how many runs were used, which columns were left out;
- the sheet checks and what each one found;
- each round: the recipes that were started and the results you recorded;
- the model checks, the inputs that went with the target, and the proposed recipes (predicted values are always called predicted);
- the last sheet Kalos could not analyse, and why;
- these docs (`docs/*.md`), and skill files (`SKILL.md`) when the engine has them.

It cannot answer anything else. It does not know about other labs, the literature, or your instruments.

## What it will not do

- It does not state anything Kalos did not record. Every sentence in an answer must point to one of your records or to the next step. A sentence that points nowhere is removed before you see it. A sentence with a number that is not in the record it points to is removed too.
- It does not promise outcomes or name causes. If an answer says a recipe *will* work, *guarantees* or *proves* something, *causes* a result, or is *optimal*, and the record it points to does not say so, the whole answer is refused.
- It does not approve anything, start runs, record results, or change a campaign. None of its routes write campaign data.
- When nothing in your data answers the question, it says so and shows the next step instead. The response then has `abstained: true`.

## Where the model runs

The **next step** and **SOP drafts** use no model, so they work on any engine.

**Ask** and **revise an SOP draft** need a model. You choose it with `KALOS_LLM_PROVIDER`:

| Setting | What happens |
|---|---|
| `ollama` | A model on this machine, through [Ollama](https://ollama.com). With the default address (`http://localhost:11434`) nothing leaves the machine. This is the default in the desktop app. |
| `openai` | A model behind any OpenAI-compatible `/v1/chat/completions` server on this machine or your own host (llama.cpp's server, LM Studio, vLLM, and similar). With a loopback address nothing leaves the machine; the status route reports `local: false` if you point it elsewhere. Set `KALOS_OPENAI_BASE_URL` and `KALOS_OPENAI_MODEL` (and `KALOS_OPENAI_API_KEY` if the server needs one). |
| `anthropic` | The Anthropic API. For a question, the question and the retrieved pieces of your data are **sent off the machine** to Anthropic. For an SOP revise, the **whole draft** (every setting, value and filled-in blank in it) and your instruction are sent. Needs `ANTHROPIC_API_KEY`. |
| `none` | No model. Ask and revise answer 404 with code `assist_disabled`. The next step, the index and SOP drafts keep working. |

Unset means `ollama` in desktop mode (`KALOS_DESKTOP=1`) and `none` everywhere else. Data goes to Anthropic only when someone sets `anthropic` on purpose.

What the model receives:

- For a question: the question, the pieces of your data that were found for it (at most six), and the next step. Never the whole sheet, never another tenant's data.
- For an SOP revise: the whole draft, including every setting, value and filled-in blank, and your instruction.

If you point `KALOS_OLLAMA_URL` at another computer, your data goes to that computer. The status route then reports `local: false`.

### Two local models: fast and quality

With `ollama`, two models share the work (`kalos/providers/cascade.py`):

| Tier | Used for | Default | Licence |
|---|---|---|---|
| fast | header mapping, short labels, intent routing, plain wording of engine warnings and refusals, setup suggestions | `phi4-mini` | MIT |
| quality | answers with sources, SOP revise, summaries | `qwen2.5:7b-instruct` | Apache-2.0 |

Older engines defaulted to `llama3.1`, which is under the Llama community licence; the defaults are now MIT and Apache-2.0 models. The choice is measured, not assumed: see [Measured model choice](#measured-model-choice).

**The rule.** A model never produces a number, an interval, a verdict or a proposal. Every reply is checked against the engine's own data before anyone sees it:

- the sources and run ids it cites exist (`nonexistent_citation`);
- the column names it uses are columns of the sheet (`unknown_column`);
- every number it states equals a number the engine gave it (`mismatched_number`);
- JSON replies parse and have the expected fields (`invalid_json`, `schema`);
- no banned word, and no claim word its sources do not use (`unsupported_claim`).

A reply that fails is never shown. For a fast task, the fast model's failed reply goes to the quality model. If both fail, the engine's own offline path answers (the deterministic header mapper, the keyword router, the engine's original wording, or the next step). Every response that involved a model says which path produced it:

```json
{"provider_path": "fast>quality>offline", "offline_fallback": true, "model": null,
 "attempts": [{"tier": "fast", "model": "phi4-mini", "outcome": "invalid", "reason": "unknown_column"},
              {"tier": "quality", "model": "qwen2.5:7b", "outcome": "invalid", "reason": "invalid_json"}]}
```

`provider_path` is `fast`, `quality`, `fast>quality`, or any of them followed by `>offline`; `offline` alone means no model was asked (nothing relevant was found, or no model is configured). In an answer, sentences that failed a check are removed and counted in `checks.drop_reasons`; when no sentence is left the answer is the next step and `abstained` is true.

With `anthropic` both tiers are the same remote model, and the same checks apply. Remote models are only used when someone sets `anthropic` on purpose. Identity columns are removed before any sheet content reaches any model (`kalos/normalize/payload.py`).

| Variable | Default | Meaning |
|---|---|---|
| `KALOS_LLM_PROVIDER` | `ollama` in desktop mode, else `none` | `ollama`, `openai`, `anthropic` or `none` |
| `KALOS_OPENAI_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible server base URL (used when provider is `openai`) |
| `KALOS_OPENAI_MODEL` | unset | the model id the OpenAI-compatible server exposes; must be set to select a model |
| `KALOS_OPENAI_API_KEY` | unset | optional bearer key for the OpenAI-compatible server |
| `KALOS_OPENAI_ALLOW_REMOTE` | unset (off) | set to `1`, `true`, `yes` or `on` to allow a non-loopback OpenAI server; prompts and any bearer key go to that host |
| `KALOS_NO_PAID_CALLS` | unset (off) | `1`, `true`, `yes` or `on` blocks Anthropic and remote OpenAI calls, even with credentials or remote access enabled; local servers still work |
| `KALOS_OPENAI_TIMEOUT_S` | `120` | how long to wait for an answer |
| `KALOS_OLLAMA_URL` | `http://localhost:11434` | where Ollama listens |
| `KALOS_LLM_FAST_MODEL` | `phi4-mini` | the fast Ollama model |
| `KALOS_LLM_QUALITY_MODEL` | `qwen2.5:7b-instruct` | the quality Ollama model |
| `KALOS_LLM_CASCADE` | `1` | `0`: a fast task uses the fast model only, then the offline path |
| `KALOS_OLLAMA_MODEL` | unset | older setting. Set on its own, it is the only model (both tiers, no cascade); set with `KALOS_LLM_FAST_MODEL`, it is the quality model |
| `KALOS_OLLAMA_TIMEOUT_S` | `120` | how long to wait for an answer |
| `KALOS_ASSIST_ANTHROPIC_MODEL` | `claude-sonnet-5` | the Anthropic model to use |
| `KALOS_ASSIST_EMBED` | `1` | `0` turns off local embeddings for the question index |
| `KALOS_EMBED_MODEL` | `nomic-embed-text` | the Ollama embedding model for semantic search |
| `KALOS_SEMANTIC_SEARCH` | `1` | `0`: campaign search is keyword search only |
| `KALOS_EMBED_CACHE_MAX` | `20000` | most vectors kept per tenant |
| `KALOS_ASSIST_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | local embedding model, used only if already on disk |
| `KALOS_ASSIST_DOCS_DIR` | the repo's `docs/` | docs to include in the index |
| `KALOS_ASSIST_SKILLS_DIR` | unset | an extra folder of `<slug>/SKILL.md` files to include |
| `KALOS_ASSIST_INDEX_ON_WRITE` | `1` | `0` stops the index refresh after uploads and results |

If the model does not answer, Ask returns 503 with code `assist_provider_unavailable`. Nothing else is affected.

## Run a local Qwen model

Set `KALOS_NO_PAID_CALLS=true` to prevent billable model calls. Anthropic
assistant chat refuses with a message naming the flag, and its status reports
`available: false`. Anthropic normalization uses the offline plan without
contacting the API. Remote OpenAI-compatible chat and probes are also blocked.
Loopback OpenAI servers and Ollama are unaffected (including remote Ollama URLs).
The flag is read on each call; unset or any value other than `1`, `true`, `yes`
or `on` leaves paid calls enabled. Values ignore case and surrounding spaces.

Install [Ollama](https://ollama.com) and start its server. Then:

```sh
ollama pull qwen2.5:7b-instruct
export KALOS_LLM_PROVIDER=ollama
python -m kalos.providers.check
```

The assistant and normalization share this default. You can pull another model
and set `KALOS_OLLAMA_MODEL=qwen3:8b`, for example. Thinking blocks are removed
from local replies before showing text or reading normalization JSON.

The check prints provider, URL, model, local, reachable and model pulled. It
suggests `ollama pull` if the server answers but the model is missing.
`GET /api/assist/index/status?probe=true` includes `reachable` and `model_present`
for local providers. The probe has a two-second timeout and runs only with this
query parameter, never at import. Ordinary status requests and provider slot
status (`/api/providers`) do not probe. The check CLI always probes.

For LM Studio or llama.cpp, start its OpenAI-compatible server, set
`KALOS_LLM_PROVIDER=openai`, `KALOS_OPENAI_BASE_URL=http://localhost:8080/v1`
(adjust the port), and `KALOS_OPENAI_MODEL` to the served model id. Set
`KALOS_OPENAI_API_KEY` only if needed. This is an assistant-only path;
normalization uses the offline mapper with `openai` selected.

## The index

To answer a question, the assistant first finds the pieces of your data that match it. It keeps a small search index for that, per tenant, in `$KALOS_STATE_DIR/assist/<tenant>/`.

- It is built the first time you ask, and when you press Rebuild.
- After an upload, a recorded result or a re-analysis, only the campaign that changed is updated.
- It finds matches by shared words (BM25). If the `assist-embed` extra is installed and its model files are already on the machine, it uses those instead. Nothing is ever downloaded.
- It is derived data. Deleting the folder loses nothing; it is rebuilt from the campaign record.

Install: `uv sync --extra assist` (BM25 and the .docx export). The `assist-embed` extra is optional.

## Semantic search

`GET /api/campaigns/{id}/search?q=...&k=10` (read scope) ranks one campaign's own records for a query: each started run (recipe, measured result, the reason given when it was started), each row of the uploaded sheet, each column and what Kalos did with it, the reason recorded with each corrected result, and the assistant index chunks for the campaign (sheet, findings, rounds, model checks, proposals). Nothing is generated: every hit is a record quoted from the campaign, so there is nothing to validate but the order.

- With `KALOS_LLM_PROVIDER=ollama`, records are ranked by meaning with a small local embedding model, [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) (Apache-2.0; `ollama pull nomic-embed-text`). `retriever` is `embeddings`.
- Otherwise, or when the model is not pulled or does not answer, records are ranked by shared words (BM25). `retriever` is `keyword` and `fallback_reason` says why. Search text is never sent to a remote provider.
- Identity columns (operator, sample name, ...) are removed from rows and recipes before anything is embedded, by the same header screen the header mapping uses.
- Bounds: a query of at most 500 characters, `k` at most 25, at most 3000 records per campaign (1000 sheet rows), each clipped to 1000 characters; 20 searches a minute per user (`KALOS_ASSIST_ASK_PER_MINUTE`, counted apart from questions).
- Vectors are cached per tenant in `$KALOS_STATE_DIR/assist/<tenant>/embeddings.sqlite`, keyed by the sha256 of the model and the text, so a record is embedded once; the cache keeps at most `KALOS_EMBED_CACHE_MAX` vectors (default 20000) and drops the least recently used. Deleting it loses nothing.

Response: `{campaign_id, retriever, model, fallback_reason, n_documents, hits[{kind, id, title, text, score}]}`. `score` is a similarity (embeddings) or a BM25 score (keyword); it orders hits and means nothing else.

## Setup suggestions

`POST /api/campaigns/{id}/setup/suggestions` `{target?}` (read scope) suggests, for each column of the campaign's uploaded sheet, a role and a plausible unit, each with a short reason and a confidence. The model helps; you decide.

- **Suggestions only.** Nothing is applied: the route writes nothing, and no part of the model fit reads a suggestion, so a suggestion can never become a prior (`tests/test_setup_suggestions.py` checks both). Every suggestion has `requires_confirmation: true`. To use one, confirm it in the campaign setup or post the roles yourself; `roles_draft` is the suggestions in the roles shape, for you to review.
- **Roles** match the analysis's column roles one to one:

| Role | Roles key | Meaning |
|---|---|---|
| `setting` | `features` (`categoricals` for a text column) | a process input you choose before the run |
| `outcome` | `target` (the one you chose), else `outputs` | a result measured after the run |
| `covariate` | `covariates` | recorded about the run but not chosen (passage, thaw) |
| `batch_group` | `groups` | a batch, lot, plate or replicate group |
| `clone` | `strain` | the production clone or strain label |
| `id` | `ids` | an identifier or free text, not used by the model |
| `derived_from_target` | `outputs` (reason `target_leak`) | computed from the result (a grade, a tier), never an input |

- **What the model sees:** each header, the unit the engine read from it, and summary statistics (count, missing, distinct values, min, median, max, up to five level labels of a short text column). Never a row, never an identity column, never free text.
- **What the model may say** is checked like every other reply: the column exists, the role is one of the seven, the unit is a known unit or none, and the reason names no other column and states no number the statistics do not hold, with no claim words. A failed reply goes to the quality model, then to the engine's own suggestion (`reason_source: engine`, `provider_path: ...offline`).
- **Confidence is the engine's, not the model's:** `high` when the engine's own evidence (the header unit, the identifier rule, the clone and covariate name rules, the target-leak check on the data) agrees with the role, `low` when it disagrees (the evidence and `engine_role` are listed), `medium` when there is none. A unit read from the header always wins over a model's unit (`unit_source`).
- Works with no model (`KALOS_LLM_PROVIDER=none`): the engine's suggestions, `provider_path: offline`. Answers 409 `no_sheet` before an upload.

## SOP drafts

A draft is written from one set of recipes: some started runs, some proposals, or every run of one round. There are three templates: run recipe (template id `batch_recipe`), media preparation, and custom.

- Every number in the draft comes from the campaign record. Each one is listed in `fields_from_record` with the record it came from.
- Predicted values are labelled Predicted. Measured results are labelled Measured.
- Everything the lab must decide is a blank, written `[[TO FILL: name]]`: safety notes, hazards, equipment, quantities to weigh out, lot numbers, acceptance limits, who prepares, checks and approves. Kalos never fills these in.
- The first line of every draft is: *Draft prepared with Kalos from campaign records on (date). Not reviewed. Not an approved procedure.* No revision can remove it.
- You fill blanks yourself. Each fill is saved as a new version with your name.
- A model can reword the prose ("revise"). The result is checked. If it changed a number, a table row, a blank, or the first line, it is refused and nothing is saved.
- Versions are never overwritten. Export as `.md`, or `.docx` with the `assist` extra.

## Routes

All routes use the caller's tenant. Read scope unless marked.

| Route | What it returns |
|---|---|
| `GET /api/assist/next?campaign_id=` | `{state, step, why, why_it_matters, action_route, blockers[{code, text, why}], campaign_id}` |
| `POST /api/assist/ask` `{question, campaign_id?}` | `{answer, sources[{kind, id, quote, campaign_id}], abstained, next, provider_path, offline_fallback, model, attempts, checks?}` |
| `GET /api/assist/index/status` | `{provider, model, local, fast_model?, quality_model?, escalation?, chunks, built_at, building, retriever}` |
| `POST /api/assist/index/rebuild` (write) | the status. Writes an audit row. |
| `POST /api/campaigns/{id}/setup/suggestions` `{target?}` | `{campaign_id, status, applied: false, notice, target, role_vocabulary, suggestions[{column, role, roles_key, unit, unit_source, reason, reason_source, confidence, evidence, engine_role, requires_confirmation}], roles_draft, confidence_basis, provider_path, offline_fallback, model, attempts}` |
| `GET /api/campaigns/{id}/search?q=&k=` | `{campaign_id, retriever, model, fallback_reason, n_documents, hits[{kind, id, title, text, score}]}` |
| `POST /api/assist/sop/draft` (write) | a draft, version 1. Writes an audit row. |
| `POST /api/assist/sop/{id}/revise` (write) | the next version, or 422 `revision_refused` |
| `POST /api/assist/sop/{id}/fill` (write) | the next version, or 400 `unknown_placeholder` |
| `GET /api/assist/sop`, `GET /api/assist/sop/{id}`, `GET /api/assist/sop/{id}/export` | the list, one version, a file |

The next-step states are `no_sheet`, `fit_refused`, `sheet_errors`, `runs_awaiting`, `runs_all_recorded`, `not_trusted`, `sheet_warnings`, `recipes_proposed`, `round_complete` and `no_recipes`. (Earlier engines called the last two `batch_proposed` and `no_batch`.) Exactly one applies to any campaign; `tests/test_assist_guide.py` checks every combination.

The plain sentences behind `why_it_matters` and each blocker's `why` are the same ones the web app shows. They live in `kalos/assist/why.py`.

## Measured model choice

The defaults above are the models the evaluation in `benchmarks/llm/` chose on a fixed test set. The table and how to rerun it are in `benchmarks/llm/RESULTS.md`.
