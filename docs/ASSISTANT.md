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

| Variable | Default | Meaning |
|---|---|---|
| `KALOS_LLM_PROVIDER` | `ollama` in desktop mode, else `none` | `ollama`, `openai`, `anthropic` or `none` |
| `KALOS_OPENAI_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible server base URL (used when provider is `openai`) |
| `KALOS_OPENAI_MODEL` | unset | the model id the OpenAI-compatible server exposes; must be set to select a model |
| `KALOS_OPENAI_API_KEY` | unset | optional bearer key for the OpenAI-compatible server |
| `KALOS_OPENAI_TIMEOUT_S` | `120` | how long to wait for an answer |
| `KALOS_OLLAMA_URL` | `http://localhost:11434` | where Ollama listens |
| `KALOS_OLLAMA_MODEL` | `llama3.1` | the Ollama model to use |
| `KALOS_OLLAMA_TIMEOUT_S` | `120` | how long to wait for an answer |
| `KALOS_ASSIST_ANTHROPIC_MODEL` | `claude-sonnet-5` | the Anthropic model to use |
| `KALOS_ASSIST_EMBED` | `1` | `0` turns off local embeddings |
| `KALOS_ASSIST_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | local embedding model, used only if already on disk |
| `KALOS_ASSIST_DOCS_DIR` | the repo's `docs/` | docs to include in the index |
| `KALOS_ASSIST_SKILLS_DIR` | unset | an extra folder of `<slug>/SKILL.md` files to include |
| `KALOS_ASSIST_INDEX_ON_WRITE` | `1` | `0` stops the index refresh after uploads and results |

If the model does not answer, Ask returns 503 with code `assist_provider_unavailable`. Nothing else is affected.

## The index

To answer a question, the assistant first finds the pieces of your data that match it. It keeps a small search index for that, per tenant, in `$KALOS_STATE_DIR/assist/<tenant>/`.

- It is built the first time you ask, and when you press Rebuild.
- After an upload, a recorded result or a re-analysis, only the campaign that changed is updated.
- It finds matches by shared words (BM25). If the `assist-embed` extra is installed and its model files are already on the machine, it uses those instead. Nothing is ever downloaded.
- It is derived data. Deleting the folder loses nothing; it is rebuilt from the campaign record.

Install: `uv sync --extra assist` (BM25 and the .docx export). The `assist-embed` extra is optional.

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
| `POST /api/assist/ask` `{question, campaign_id?}` | `{answer, sources[{kind, id, quote, campaign_id}], abstained, next}` |
| `GET /api/assist/index/status` | `{provider, model, local, chunks, built_at, building, retriever}` |
| `POST /api/assist/index/rebuild` (write) | the status. Writes an audit row. |
| `POST /api/assist/sop/draft` (write) | a draft, version 1. Writes an audit row. |
| `POST /api/assist/sop/{id}/revise` (write) | the next version, or 422 `revision_refused` |
| `POST /api/assist/sop/{id}/fill` (write) | the next version, or 400 `unknown_placeholder` |
| `GET /api/assist/sop`, `GET /api/assist/sop/{id}`, `GET /api/assist/sop/{id}/export` | the list, one version, a file |

The next-step states are `no_sheet`, `fit_refused`, `sheet_errors`, `runs_awaiting`, `runs_all_recorded`, `not_trusted`, `sheet_warnings`, `recipes_proposed`, `round_complete` and `no_recipes`. (Earlier engines called the last two `batch_proposed` and `no_batch`.) Exactly one applies to any campaign; `tests/test_assist_guide.py` checks every combination.

The plain sentences behind `why_it_matters` and each blocker's `why` are the same ones the web app shows. They live in `kalos/assist/why.py`.
