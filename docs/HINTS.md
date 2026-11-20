# Hints

Three levels for every stop. Try level 1 first, read level 2 when you are stuck, level 3 is almost the code (never the
whole solution). Run `python -m pytest -k t0N` after each change; the failing line tells you what is missing.

## T01 - the Query schema and the parse chain (`src/factdesk/llm/query.py`)

1. **Idea.** The model must not answer; it must only read the question into four boxes. A schema with descriptions is
   the form it fills in; a chain is "prompt, then model that fills the form". The prompt text is already written in
   `llm/prompts.py` (`PARSE_SYSTEM`, `PARSE_HUMAN`): read it, it tells you what each field means.
2. **API.** `pydantic.BaseModel` with `Field(description=...)`; `typing.Literal[...]` for a fixed list of values;
   `ChatPromptTemplate.from_messages([("system", ...), ("human", ...)])`; `chat.with_structured_output(Query)`;
   the `|` operator joins a prompt and a model into a Runnable.
3. **Near-code.**
   ```python
   operation: Literal["value", "compare", ...] = Field(description="...")
   indicator_text: str = Field(description="...")      # likewise country_texts: list[str], years: list[int]
   prompt = ChatPromptTemplate.from_messages([("system", PARSE_SYSTEM), ("human", PARSE_HUMAN)])
   return prompt | chat.with_structured_output(Query)
   ```
   Four fields only: a 1.5B model fills small flat schemas; an enum is safer than free text.

## T02 - the `get_value` and `growth` tools (`src/factdesk/agents/tools.py`)

1. **Idea.** A tool is a function the model may call. The model reads only the NAME, the TYPED arguments and the
   DOCSTRING to decide when and how. The arithmetic is not yours to write: `run_tool` does it and returns JSON text,
   also for errors (an exception would end the agent; an error as JSON lets it try again).
2. **API.** `@tool` from `langchain.tools`; the docstring is the description the model sees; look at the given
   `compare` tool and copy its shape. `run_tool(table, numbers, operation=..., indicator=..., countries=[...], years=[...])`.
3. **Near-code.**
   ```python
   @tool
   def get_value(country: str, indicator: str, year: int) -> str:
       """one line what it does ... Input: ... Output: ..."""
       return run_tool(table, numbers, operation="value", indicator=indicator, countries=[country], years=[year])
   ```
   `growth` is the same with `operation="growth"` and `years=[year_from, year_to]`. Check the argument names with
   `tools["growth"].args`: the test expects `country, indicator, year_from, year_to`.

## T03 - `link_country` (`src/factdesk/knowledge/linker.py`)

1. **Idea.** Names in questions are messy ("S. Korea", "Korea", "Rep. of Korea"). Cheap and certain first: an exact
   match on a normalised alias table. Only when that fails, use meaning: search the vector database, restricted to
   country chunks. Then decide: is the best candidate clearly the best (link), nearly tied with another country or only
   a weak guess (ask the user), or nothing close (unknown)?
2. **API.** `normalize(text)` and the alias `index` (key -> list of codes; two codes = a shared name);
   `store.search(text, k=graph.link_k, where={"kind": "country"})` returns `Hit`s with `.score` and
   `.metadata["code"]`; the result is a `Link(text, status, code, method, candidates)` of `Candidate(code, name, score)`.
3. **Near-code.**
   ```python
   codes = index.get(normalize(text), [])
   if codes: ...                      # one code: linked (score 1.0, method "alias"); several: ambiguous
   best = {}
   for hit in store.search(...): best[hit.metadata["code"]] = max(hit.score, best.get(..., 0.0))
   ranked = sorted(best.items(), key=lambda item: -item[1])
   # unknown if no hit or ranked[0][1] < graph.country_min_score
   # close = every country with top - score < graph.ambiguity_margin ; ambiguous if len(close) > 1
   #         or top < graph.country_confident_score ; else linked
   ```
   Try `python main.py link "S. Korea"`, `link "Korea"`, `link "Rep. of Korea"`, `link "Taiwan"` and read the scores.

## T04 - the three routers (`src/factdesk/graph/routes.py`)

1. **Idea.** A router is a pure function: state in, NAME of the next node out. No model, no side effect. The nodes
   already leave what the router needs in the state: `link` sets `status` when it declines and fills `pending` when a
   name is ambiguous; `compute` sets `status`; `verify` sets `verified`; `explain` counts `tries`.
2. **API.** Plain dict access (`state.get("status")`); `END` from `langgraph.graph` (already imported) is the name
   that ends the graph.
3. **Near-code.**
   ```python
   if state.get("status"): return "decline"       # after_link: then pending -> "clarify", else "compute"
   return "explain" if state["status"] == "answered" else "decline"             # after_compute
   if state["verified"]: return END                                              # after_verify
   return "explain" if state["tries"] < state["max_tries"] else "template"
   ```
   Draw the graph from the README on paper first: each router is one diamond.

## T05 - the clarify node (`src/factdesk/graph/nodes/clarify.py`)

1. **Idea.** An `interrupt` stops the graph and waits for a person; the graph is saved in the checkpoint. When you
   resume, the node runs AGAIN from its first line and `interrupt(...)` now returns the person's answer. So anything
   you do before `interrupt` happens twice: the given `notify_once` makes your outside effect (the outbox line) safe.
2. **API.** `from langgraph.types import interrupt`; `interrupt(value)` returns the resume value; `notify_once(outbox,
   key=..., payload=...)`; the node returns a partial state update (a dict). `pending[0]` is
   `{"text": ..., "candidates": [{"code", "name", "score"}, ...]}`.
3. **Near-code.**
   ```python
   ask = state["pending"][0]
   notify_once(outbox, key=f"{state['thread_id']}:{ask['text']}", payload={...})
   choice = interrupt({"question": ..., "text": ask["text"], "options": ask["candidates"]})
   # refuse a choice that is not one of the candidate codes (ValueError), then
   return {"resolved": {**state.get("resolved", {}), ask["text"]: choice}, "pending": [], "log": [...]}
   ```
   `resolved` has no reducer: copy the old dict and add yours, or you lose earlier choices.

## T06 - `result_exact` and `answer_numbers_verified` (`src/factdesk/evaluation/evaluators.py`)

1. **Idea.** An evaluator is a function that returns True/False (or a number) for ONE case, with named parameters
   `inputs, outputs, reference_outputs`. No model judges here: the expected number comes from the data, and "no invented
   numbers" is exactly what the `verify` step already checks, so the same function can judge v1 too.
2. **API.** `outputs["result"]` is `{"label", "value"}` or `None`; `reference_outputs["result_value"]` is a number or
   `None`; `EvalConfig().result_rel_tol`; `find_unverified(answer, facts=..., result=..., years=..., numbers=FormatConfig())`
   returns the list of bad numbers. A correct hand-over repeats the year of the question, so add the years written in
   `inputs["question"]` (`re.findall(r"\b(?:19|20)\d\d\b", ...)`) to the years the answer may contain.
3. **Near-code.**
   ```python
   want = reference_outputs["result_value"]; result = outputs.get("result")
   if want is None: return result is None
   ...  return abs(result["value"] - want) <= tolerance * abs(want)
   asked = [int(year) for year in re.findall(r"\b(?:19|20)\d\d\b", inputs["question"])]
   bad = find_unverified(outputs.get("answer", ""), facts=outputs.get("facts", []), ..., years=outputs.get("years", []) + asked, ...)
   return not bad
   ```

## T07 - the labels of a case (`src/factdesk/observability/trace_config.py`) and `my_work/notes.toml`

1. **Idea.** A trace you cannot find is useless. Give each case run a name, tags and metadata: then "all v1 runs" or
   "case D06 in both versions" is a filter, locally and on LangSmith. Then READ traces: that is where you find the agent's
   wrong argument and the case the verify step saved.
2. **API.** The given `run_config(tracing, tracer=..., run_name=..., metadata=..., extra_tags=...)` does the work; you
   decide the labels. Read traces with `python main.py trace-show runtime/traces/<file>.json`; `ls runtime/traces`.
3. **Near-code.**
   ```python
   return run_config(tracing, tracer=tracer, run_name=f"{version}-{case_id}",
                     metadata={"version": version, "case_id": case_id, "backend": backend}, extra_tags=(version,))
   ```
   For the notes: in a v1 trace open the `tool` runs and compare the arguments with the question (a country code, a
   year, an indicator). For "verify saved": in a v2 trace look for TWO `explain` runs (the retry) or a `template` run, and read the
   `verify` line before them. If you find no such case in the dev set, say so in the notes and name the closest
   thing (e.g. a case where the model rounded a number) - do not invent one.

## T08 - `compare_answers` and the Langflow flow (`src/factdesk/integrations/flow_compare.py`, `flows/`)

1. **Idea.** Two answers to one question: the code's and the flow's. Compare them by RULES, not by reading: does each
   contain the words a right answer needs, and, for a numeric question, are its numbers the table's? The flow has no
   code step, so on the numeric question it can only quote or guess: your comparison must show that.
2. **API.** `find_unverified(text, facts=..., result=..., years=..., numbers=...)`; `AnswerComparison(...)` has four lists.
   `facts=None` means "a definition question: do not check numbers".
3. **Near-code.**
   ```python
   missing = [term for term in expected_terms if term.lower() not in answer.lower()]   # for each answer
   unverified = [] if facts is None else find_unverified(answer, facts=facts, ...)
   return AnswerComparison(code_missing_terms=..., flow_missing_terms=..., code_unverified=..., flow_unverified=...)
   ```
   Building the flow: use the values in `configs/app.toml` section `[rag]` and the README table; export with the
   Langflow `Export` menu into `flows/`, and check it with `python main.py flow-show flows/<file>.json` before you submit
   (it must say `[OK] no password field holds a value`).

**Langflow 1.12.6 (checked on a real run).** UI names: Chat Input, OpenAI Embeddings, Chroma DB (Persist Directory EMPTY: it talks to the server by host/port), Parser, Prompt Template, OpenAI, Chat Output. The model dropdowns list only OpenAI names: TYPE the local model name, set the base URL to the model server (`http://127.0.0.1:8000/v1`), API key `local`. Switch TikToken OFF on OpenAI Embeddings. `flow-ask` / `flow-compare` need `LANGFLOW_API_KEY` in the environment. Start model server, then `chroma run`, then `index` (server store), then Langflow. The flow has no similarity threshold: it always passes k notes, so it may answer what the notes do not cover.

## T09 - the report (`REPORT.md`)

1. **Idea.** The report says what you SAW, with numbers from your own results files, not what the tools promise.
2. **Where the facts are.** `results/v1.json`, `results/v2.json` (and the `.csv`), `python main.py compare v1 v2`,
   `runtime/traces/`, `results/flow_compare.json`.
3. **Near-text.** For each error: case id, what the system answered, what was right, which step is to blame (parse, link,
   compute, explain, verify), and how you know (trace line). For "when is an agent enough": name one thing the agent
   did well, one thing only the graph guarantees, and the cost (seconds and model calls per case).
