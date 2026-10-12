# mtRAG judging in one folder

Run commands from `judge_run`. This folder judges each turn with the project's
mtRAG-style judge and lets a person review/annotate the result. Existing run JSON
files are retained; the live server reads the current `out.mtrag.json` without
modifying prior results. This step selects failed RAG cases for later attribution;
it does not assign one of the six failure causes.

## Commands (PowerShell)

Create a new run name so previous outputs are not overwritten:

```powershell
python run.py prepare --name conversation_new --limit 3
python run.py preview --name conversation_new
```

`prepare` reads `../conversation.json` and writes
`runs/conversation_new/input.json`. For a different source, use
`python prepare_input.py --src ..\conversation1.json --out runs/conversation1`,
then continue with `--name conversation1`. The preparer preserves selected
passages, reference answers, answerability and recorded history. `preview`
writes `preview.json` without any paid API call.

Set the approved model and API key environment variables in the same session,
then run mtRAG. The key itself must be set in `OPENROUTER_API_KEY` and is not
stored in the run files:

```powershell
$env:JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
$env:JUDGE_MODEL = "qwen/qwen3-30b-a3b"
$env:JUDGE_API_KEY_ENV = "OPENROUTER_API_KEY"
python run.py mtrag --name conversation_new
python run.py serve
```

If a thinking model stops the short IDK call at `finish_reason=length`, the
IDK completion budget can be raised for that run without changing the prompts
or score rule, for example:

```powershell
python -B run.py mtrag --name conversation_vector_qwen35b --idk-max-tokens 4096
```

The default is 512. Failed IDK requests retry with twice the previous budget,
so 4096 can request up to 16384 on the third attempt and use more credits.
This option is recorded in `out.mtrag.json` and cannot be mixed with a
different IDK budget when resuming completed records. A failed record has no
saved turn-level checkpoint, so rerunning it repeats its earlier API calls.

For a run that leaves the completion limit to the provider/model, use:

```powershell
python -B run.py mtrag --name conversation_vector_qwen35b --no-max-tokens
```

This omits `max_tokens` for every mtRAG API call, including RB_llm, both RAGAS
steps, and IDK. It cannot be combined with `--idk-max-tokens`. The provider and
model still have their own output limits; reasoning tokens count toward those
limits and are billed. If a call ends with `finish_reason=length`, the judge
stops that call immediately instead of retrying the same unlimited request.
The selected mode is recorded in the output so resume cannot mix it with a
bounded run. The ordinary command without this flag keeps the prior budgets.

For an explicitly budgeted thinking-model run, keep thinking on while limiting
each request to 4096 reasoning tokens and 8192 completion tokens in total:

```powershell
python -B run.py mtrag --name conversation_new_budgeted --reasoning-max-tokens 4096 --max-tokens 8192
```

This sends `reasoning.max_tokens` and `max_tokens` to OpenRouter for every
mtRAG call, including the RAGAS and IDK steps. The 8192 completion ceiling
stays fixed across retries; a `length` response is not retried with a higher
ceiling. The judge also rejects a response if its reported reasoning-token use
exceeds 4096, or if reported completion use exceeds 8192. The output usage
records how many calls reported token counts and how many could not be verified.
Provider support must be checked with a small pilot: a provider may ignore a
reasoning budget, and a completed answer can still hit its total
completion limit. Both budgets are recorded in the result, and runs with
different budgets cannot be merged by `--resume-from`. Use a new run name;
do not overwrite the previous unbounded result.

Open `http://127.0.0.1:8765/run/conversation_new`. The live page regenerates
from the latest JSON when refreshed. Alternatively,
`python run.py annotate --name conversation_new` writes a static
`runs/conversation_new/annotate.html` once and refuses to overwrite one that
already exists. Export human annotations with the page's JSON or CSV button.
After changing `make_annotation.py`, restart `python run.py serve` so the live
server imports the new UI code; a previously generated static HTML file does
not update on refresh.
For descriptive three-band agreement with older three-level human exports, run:

```powershell
python calibrate.py --work-dir runs/conversation_new --labels annotations_conversation_new.json
```

`calibrate.py` compares historical human `human_score_band` values with the
judge's RB_llm band. The current annotation page asks for a binary
`human_label` (`correct` or `incorrect`), so new exports do not supply the
three-level human ratings needed by `calibrate.py`. It does not change the
binary failure gate or estimate the accuracy of the later Xie-vs-Proposed
failure attribution.

## Per-turn score and provenance

| Value | Inputs and calculation | Source |
| --- | --- | --- |
| `RB_llm` | The LLM receives the prior conversation, current question, reference answer, RAG answer and selected passages. It rates faithfulness, appropriateness and completeness from 1 to 10. The parsed score is also divided by 10. | [mtRAG §6.3](https://arxiv.org/html/2501.03468#S6.SS3), [IBM prompt/parser](https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/judge_utils.py). |
| `RL_F` | A RAGAS-style faithfulness step splits the answer into statements and checks each against selected passages. Score = supported statements / all statements, from 0 to 1. The reference answer is not used in this step. | [mtRAG §6.3.2](https://arxiv.org/html/2501.03468#S6.SS3), [RAGAS](https://arxiv.org/abs/2309.15217). |
| `idk` | An LLM detects whether the answer signals insufficient information: yes=1, partial=0.5, no=0. | [mtRAG §6.3.3](https://arxiv.org/html/2501.03468#S6.SS3), [IBM IDK prompt](https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/judge_utils.py). |
| `phi` | **Project adaptation:** harmonic mean of `RB_llm/10` and `RL_F`: `2 / (1/(RB_llm/10) + 1/RL_F)`. If `RL_F` is undefined, use `RB_llm/10`; if a defined component is zero, use zero. | Inspired by mtRAG metrics; [official leaderboard](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/) uses the harmonic mean of **three** metrics for system ranking. |
| `conditioned_score` | If ANSWERABLE/PARTIAL and IDK=yes, 0; otherwise `phi`. If UNANSWERABLE/CONVERSATIONAL, 1 only for IDK=yes, otherwise 0. If answerability is missing, `phi`. | mtRAG conditions **each metric individually** for answerable/partial/unanswerable turns. Combining first and treating conversational/missing answerability as above are local adaptations. |

The project's existing binary screening rule is `correct` iff
`conditioned_score >= 0.6`; this cutoff is not claimed to be the paper's.
The `RB_llm` per-metric verdict uses `>= 6`. The page also shows these
**project-only three levels on the raw RB_llm rating**: low `<=5`, medium
`>5 and <=8`, high `>8`. Exactly 5 is low; exactly 8 is medium. They
do not change `correct/incorrect` or failure-attribution eligibility.
Historical `vrag_accept_min` and `ar` keys remain in the mtRAG JSON threshold
metadata solely so old mtRAG runs can be resumed with the same schema; neither
starts a vRAG or standalone RAGAS judge.

The paper's notation `phi` denotes an individual metric among
`RB_alg`, `RB_llm` and `RL_F`; it is not the two-metric harmonic mean in
this implementation. The paper uses several judge models and their median
for RB_llm. This implementation uses the configured model once per turn,
omits RB_alg, and uses a shortened IDK few-shot prompt. Report this as an
**mtRAG-style project adaptation**, not an exact replication.

## Files and output

- `run.py`: mtRAG-only entry point.
- `prepare_input.py`: copies conversation, gold answers and selected
  context into a new run's `input.json`; no RAG regeneration.
- `judge_paper.py`: mtRAG score, including its required RAGAS faithfulness
  component, API transport and exact-configuration resume.
- `make_annotation.py`: HTML view of mtRAG and binary human annotation.
- `annotation_server.py`: local live run picker and annotation page.
- `calibrate.py`: descriptive three-level human/model agreement.
- `runs/<name>/out.mtrag.json`: per-turn scores, binary screening labels,
  reasoning and provenance. Other historical `out.*.json` files remain
  in `runs/` for audit; they are not read by the current UI.

The human export stores the current binary choice in `human_label`. It retains
`human_score_band` only for historical three-level annotations. Failure type is
selected separately from the locked six-category taxonomy. Do not equate
a quality band with a failure cause.
