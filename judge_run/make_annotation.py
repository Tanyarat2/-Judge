#!/usr/bin/env python3
"""Build a self-contained mtRAG annotation sheet for RAG turns.

    python judge_run/make_annotation.py --work-dir judge_run/runs/pilot5

Reads input.json and out.mtrag.json from the run folder. Existing result JSON is
never overwritten. The human binary label is stored separately from the
judge's binary screening label and the six-category failure annotation.
"""

import argparse
import html
import json
import pathlib

JUDGE_TITLES = {"mtrag": "mtRAG (RB_llm + RL_F + IDK)"}
# Locked six-category taxonomy first; the extra entries are annotation-only escape hatches.
TAXONOMY = ["knowledge_boundary", "chunking", "retrieval", "context_selection",
            "grounding", "response_coverage"]
EXTRA_TYPES = ["not_a_failure", "outside_taxonomy", "insufficient_evidence"]
DEFAULT_THRESHOLDS = {"f": 1.0, "mtrag": 0.6, "rb_llm_min_rating": 6.0}


def rb_band(score):
    """Project-only display bands on the raw 1-10 RB_llm rating."""
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    if score <= 5:
        return "low"
    if score <= 8:
        return "mid"
    return "high"


def load(work_dir, name):
    path = work_dir / ("out.%s.json" % name)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return {record["id"]: record for record in json.load(stream)}


def turn_texts(conversation):
    result, questions = [], []
    for message in conversation:
        if message["role"] == "user":
            questions.append(message["content"])
        elif message["role"] == "assistant":
            result.append(("\n".join(questions), message["content"]))
            questions = []
    return result


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "%.3f" % value
    return str(value)


def not_run(status="not_run", headline="not run"):
    return {"status": status, "label": None, "headline": headline, "reasoning": "",
            "metrics": [], "notes": []}


def score_steps(scores, label, band, cutoff):
    """Explain saved mtRAG scores without recalculating or changing the verdict."""
    rb = scores.get("RB_llm")
    f = scores.get("RL_F")
    phi = scores.get("phi")
    conditioned = scores.get("conditioned_score")
    idk = scores.get("idk")
    answerability = scores.get("answerability")
    if not isinstance(rb, (int, float)) or isinstance(rb, bool):
        return []
    rb_norm = rb / 10
    steps = [
        "RB_llm: the LLM reads this turn's question, previous conversation, "
        "reference answer, RAG answer and selected passages. It gives one holistic "
        "1-10 rating for faithfulness, appropriateness and completeness; these "
        "criteria have no fixed numeric weights. %s/10 = %s (project level: %s)." %
        (fmt(rb), fmt(rb_norm), band or "n/a")]
    statements = scores.get("statements") or []
    if f is None:
        steps.append("RL_F: unavailable (no factual statements were extracted).")
        steps.append("phi: RL_F is unavailable, so phi = RB_llm/10 = %s." % fmt(phi))
    else:
        supported = sum(item.get("verdict") == "yes" for item in statements)
        detail = "%d/%d supported statements = " % (supported, len(statements)) if statements else ""
        steps.append("RL_F: %s%s. Each statement is checked against the selected "
                     "passages; the reference answer is not used for this metric." %
                     (detail, fmt(f)))
        if f <= 0:
            steps.append("phi: RL_F is 0, so the harmonic mean is 0.")
        else:
            steps.append("phi: 2 / (1/%s + 1/%s) = %s. This two-metric "
                         "per-turn combination is a project adaptation." %
                         (fmt(rb_norm), fmt(f), fmt(phi)))
    if answerability is None:
        rule = "answerability is unavailable, so the score stays at phi"
    elif answerability in ("UNANSWERABLE", "CONVERSATIONAL"):
        rule = ("the question is marked %s; IDK=%s, so the score is 1 only "
                "for IDK=1, otherwise 0" % (answerability, fmt(idk)))
    else:
        rule = ("the question is marked %s; IDK=%s, so the score is 0 "
                "for IDK=1, otherwise phi" % (answerability, fmt(idk)))
    steps.append("Conditioned score: %s; %s -> %s." % (rule, fmt(phi), fmt(conditioned)))
    steps.append("Screening label: conditioned score %s %s %s -> %s. "
                 "The cutoff and three RB_llm levels are project choices." %
                 (fmt(conditioned), ">=" if conditioned >= cutoff else "<",
                  fmt(cutoff), label))
    return steps


def scored_view(turn, cutoff=DEFAULT_THRESHOLDS["mtrag"]):
    if turn is None:
        return not_run()
    if turn["label"] == "not_judged":
        return {"status": "not_judged", "label": None, "headline": "not judged (earlier failure)",
                "reasoning": turn.get("reasoning", ""), "metrics": [], "notes": []}
    scores = turn.get("scores") or {}
    verdicts = scores.get("metric_verdicts") or {}
    band = rb_band(scores.get("RB_llm"))
    headline = "RB_llm %s/10 (%s); conditioned score %s" % (
        fmt(scores.get("RB_llm")), band or "unavailable", fmt(scores.get("conditioned_score")))
    metrics = [
        ["RB_llm (1-10)", fmt(scores.get("RB_llm")), verdicts.get("RB_llm") or ""],
        ["RB_llm level (project bins)", band or "n/a", ""],
        ["RL_F (faithfulness)", fmt(scores.get("RL_F")), verdicts.get("RL_F") or ""],
        ["IDK judge", fmt(scores.get("idk")), verdicts.get("IDK") or ""],
        ["answerability", fmt(scores.get("answerability")), ""],
        ["phi (harmonic mean)", fmt(scores.get("phi")), ""],
        ["conditioned score", fmt(scores.get("conditioned_score")), verdicts.get("conditioned") or ""]]
    notes = ["[%s] %s" % (item.get("verdict", "?"), item.get("statement", ""))
             for item in scores.get("statements") or []]
    return {"status": "judged", "label": turn["label"], "headline": headline,
            "score_band": band, "reasoning": turn.get("reasoning", ""),
            "metrics": metrics, "notes": notes,
            "score_steps": score_steps(scores, turn["label"], band, cutoff)}


def find_turn(record, turn):
    if record is None:
        return None
    for item in record.get("evaluation", {}).get("turn_evaluations", []):
        if item["turn"] == turn:
            return item
    return None


def load_input_records(work_dir):
    path = work_dir / "input.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return {record["id"]: record for record in json.load(stream)}


def build_data(work_dir):
    work_dir = pathlib.Path(work_dir)
    inputs = load_input_records(work_dir)
    judged = load(work_dir, "mtrag")
    if not inputs and not judged:
        raise SystemExit("No input.json or out.mtrag.json found in %s" % work_dir)
    ids = list(inputs)
    for cid in judged or {}:
        if cid not in ids:
            ids.append(cid)
    thresholds = dict(DEFAULT_THRESHOLDS)
    for record in (judged or {}).values():
        thresholds.update({k: v for k, v in record.get("judge", {}).get("thresholds", {}).items()
                           if k in thresholds and v is not None})
        break
    rows = []
    for cid in ids:
        source = inputs.get(cid)
        if source is None:
            source = judged[cid].get("input", judged[cid])
        texts = turn_texts(source["conversation"])
        golds = source.get("ground_truth_answers") or []
        contexts = source.get("contexts")
        answerability = source.get("answerability")
        for k, (question, answer) in enumerate(texts, 1):
            if judged is None:
                view = not_run()
            elif cid not in judged:
                view = not_run("record_missing", "record missing from output")
            else:
                turn = find_turn(judged[cid], k)
                view = scored_view(turn, thresholds["mtrag"]) if turn is not None else not_run(
                    "turn_missing", "turn missing from output")
            selected = None if contexts is None or k - 1 >= len(contexts) else contexts[k - 1]
            rows.append({
                "key": "%s#%d" % (cid, k), "id": cid, "turn": k, "n_turns": len(texts),
                "question": question, "answer": answer,
                "gold": golds[k - 1] if k - 1 < len(golds) else "",
                "answerability": (str(answerability[k - 1]).upper()
                                  if answerability and k - 1 < len(answerability) and answerability[k - 1]
                                  else "not recorded"),
                "contexts": None if selected is None else list(selected),
                "judges": {"mtrag": view},
            })
    coverage = {"mtrag": {
        "records": len(judged or {}),
        "turns": sum(row["judges"]["mtrag"]["status"] == "judged" for row in rows),
        "source": "out.mtrag.json" if judged is not None else None}}
    return {"run": work_dir.name, "rows": rows, "judge_titles": JUDGE_TITLES,
            "failure_types": TAXONOMY, "extra_types": EXTRA_TYPES, "thresholds": thresholds,
            "coverage": coverage, "total_records": len(ids),
            "criteria_html": criteria_html(thresholds)}


def criteria_html(thresholds):
    """Explain the source of each score and distinguish paper from project choices."""
    cutoff = html.escape(fmt(thresholds["mtrag"]))
    return (
        "<section><h3>mtRAG-style evaluation in this project</h3>"
        "<p><b>RB_llm (1–10):</b> An LLM judge reads the current question, previous "
        "conversation, reference answer, RAG answer, and selected passages. It rates "
        "faithfulness, appropriateness, and completeness with the mtRAG prompt adapted "
        "from RAD-Bench. The code parses Rating: [[n]] and stores both n and n/10. "
        "This is one holistic rating; the three criteria have no fixed numeric "
        "weights, and 7/10 does not mean seven of ten claims are correct. "
        "The paper used several judge models and their median; this run uses the "
        "configured model once per turn.</p>"
        "<p><b>RL_F (0–1):</b> RAGAS faithfulness splits the answer into statements, "
        "checks each against selected passages, then divides supported statements by "
        "all statements. It does not consult the reference answer.</p>"
        "<p><b>IDK:</b> The LLM decides whether the answer says there is insufficient "
        "information: yes=1, partial=0.5, no=0. The local prompt uses fewer examples "
        "than the IBM implementation.</p>"
        "<p><b>phi (0–1):</b> Here phi = 2 / (1/(RB_llm/10) + 1/RL_F), the harmonic "
        "mean of two metrics when RL_F is defined and both are positive; otherwise "
        "it is 0 if either is 0, or RB_llm/10 when RL_F is undefined. This two-metric "
        "per-turn composite is <b>our adaptation</b>. In the mtRAG paper, phi denotes "
        "one of the individual metrics RB_alg, RB_llm, or RL_F. The mtRAG leaderboard "
        "uses a harmonic mean of all three metrics for system ranking.</p>"
        "<p><b>Conditioned score (0–1):</b> If answerability is ANSWERABLE/PARTIAL "
        "and IDK=yes, score=0; otherwise score=phi. If UNANSWERABLE/CONVERSATIONAL, "
        "score=1 only when IDK=yes, otherwise 0. If answerability is absent, score=phi. "
        "The paper describes conditioning each metric separately for answerable, partial, "
        "and unanswerable turns; the conversational and unknown cases are local choices. "
        "The judge's existing binary screening label is <b>correct iff conditioned "
        "score ≥ " + cutoff + "</b>; this cutoff is project-set.</p>"
        "<p><b>Three levels on this page:</b> RB_llm ≤5 = low; &gt;5 to ≤8 = medium; "
        "&gt;8 = high. These are project display/annotation bands, not thresholds from "
        "the mtRAG paper and not the binary failure gate.</p>"
        "<p>Sources: <a href='https://arxiv.org/html/2501.03468#S6.SS3' "
        "target='_blank' rel='noreferrer'>mtRAG paper §6.3</a>; "
        "<a href='https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/judge_utils.py' "
        "target='_blank' rel='noreferrer'>IBM judge prompt and rating parser</a>; "
        "<a href='https://ibm.github.io/mt-rag-benchmark/MTRAGEval/' "
        "target='_blank' rel='noreferrer'>official leaderboard formula</a>.</p>"
        "<p>Judge scores screen RAG answers; a human should verify failed turns "
        "before failure attribution.</p></section>"
    )


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>mtRAG annotation</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f5f7;color:#1b1f23}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccd;padding:8px 16px;z-index:5}
header label{margin-right:12px;font-size:13px}
main{max-width:1100px;margin:12px auto;padding:0 12px}
.card{background:#fff;border:1px solid #d5d8de;border-radius:6px;margin:12px 0;padding:12px}
.card.done{border-left:6px solid #2a9d4b}
.card>summary{cursor:pointer;list-style:none}.card>summary::-webkit-details-marker{display:none}
.card>summary:before{content:'+';display:inline-grid;place-items:center;width:18px;height:18px;border:1px solid #aab;border-radius:3px;font-family:monospace}
.card[open]>summary:before{content:'-'}
.hd{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}
.badge{font-size:12px;padding:2px 7px;border-radius:10px;background:#e3e5e8}
.correct,.high{background:#cdeed6}.incorrect,.low{background:#f6c9c9}.mid{background:#fff0c2;color:#785500}.not_judged,.not_run,.record_missing,.turn_missing{background:#e3e5e8;color:#666}
#coverage{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}#coverage .badge{border:1px solid #cbd0d6}
.blk{margin:6px 0;white-space:pre-wrap}.blk b{display:block;font-size:12px;color:#555}
details{margin:4px 0}summary{cursor:pointer;font-size:13px}
.judge{border-top:1px dashed #ccd;margin-top:6px;padding-top:4px}
.calc{margin:8px 0;padding:8px 12px;background:#f5f8fc;border-left:3px solid #5979a6;border-radius:3px;font-size:13px;line-height:1.5}
.calc ol{margin:6px 0 0;padding-left:22px}.calc li{margin:4px 0}
table.m td{padding:1px 8px 1px 0;font-size:13px}
.ann{background:#f0f4ff;border-radius:4px;padding:8px;margin-top:8px}
.ann select,.ann textarea,.ann input{font-size:13px}
.ann textarea{width:98%;height:48px}
#guidelines{line-height:1.5}#guidelines>summary{font-weight:700;font-size:16px}
.guide-body{padding:4px 10px}.guide-body p{margin:9px 0}
.case{margin:14px 0;border-top:1px solid #ccd}.case>summary{position:sticky;top:83px;z-index:2;display:flex;flex-wrap:wrap;gap:7px;align-items:center;padding:9px 4px;background:#f4f5f7;font-weight:700;list-style:none}
.case>summary::-webkit-details-marker{display:none}.case>summary:before{content:'+';display:inline-grid;place-items:center;width:20px;height:20px;border:1px solid #aab;border-radius:3px;font-family:monospace}
.case[open]>summary:before{content:'-'}
.case-turns{padding:2px 0 10px 12px;border-left:2px solid #d5d8de}
.blind .judges{display:none}
</style></head><body>
<header>
<b id="title"></b> &nbsp; <span id="progress"></span><br>
<label>Annotator <input id="annotator" size="10"></label>
<label><input type="checkbox" id="fTodo"> unlabelled only</label>
<label><input type="checkbox" id="fFail"> mtRAG says incorrect</label>
<label><input type="checkbox" id="blind"> blind (hide judge results)</label>
<button id="expJson">Export JSON</button> <button id="expCsv">Export CSV</button>
<label>Import JSON <input type="file" id="imp" accept=".json"></label>
<a href="#guidelines">How mtRAG scores work</a>
</header>
<main>
<div id="coverage" aria-label="Judge output coverage"></div>
<details id="guidelines" class="card" open><summary>How mtRAG scores work — RB_llm, phi and conditioned score</summary><div id="crit" class="guide-body"></div></details>
<div id="cards"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const KEY = 'annot:' + D.run;
let A = JSON.parse(localStorage.getItem(KEY) || '{}');
// Earlier binary UI revisions saved correct/incorrect in the three-level field.
// Move only those values; preserve genuine low/mid/high annotations for export.
function migrateBinaryLabel(a){
  if(a.human_score_band!=='correct' && a.human_score_band!=='incorrect') return false;
  if(a.human_label!=='correct' && a.human_label!=='incorrect') a.human_label=a.human_score_band;
  delete a.human_score_band;
  return true;
}
let migrated = false;
Object.values(A).forEach(a=>{if(migrateBinaryLabel(a)) migrated=true;});
if(migrated) localStorage.setItem(KEY, JSON.stringify(A));
const $ = id => document.getElementById(id);
function el(tag, cls, text){const e=document.createElement(tag); if(cls)e.className=cls; if(text!==undefined)e.textContent=text; return e;}
function save(){localStorage.setItem(KEY, JSON.stringify(A)); progress();}
function done(r){const a=A[r.key]; return !!(a && (a.human_label==='correct' || a.human_label==='incorrect'));}
function progress(){
  const n=D.rows.filter(done).length; $('progress').textContent=n+' / '+D.rows.length+' labelled';
  document.querySelectorAll('.card[data-key]').forEach(c=>c.classList.toggle('done', done(D.rows.find(r=>r.key===c.dataset.key))));
}
function badge(name, v){
  const score=v.score_band?' | RB_llm: '+v.score_band:'';
  return el('span','badge '+(v.score_band||v.label||v.status),
            'mtRAG: '+(v.label||v.status.replace('_',' '))+score);
}
function block(label, text){const d=el('div','blk'); d.appendChild(el('b',null,label)); d.appendChild(document.createTextNode(text||'')); return d;}
function judgeView(name, v){
  const d=el('details','judge'); const s=el('summary',null,D.judge_titles[name]+' -> '+(v.display_label||v.label||v.status)+' ('+v.headline+')'); d.appendChild(s);
  if(v.reasoning) d.appendChild(block('reasoning', v.reasoning));
  if(v.metrics.length){const t=el('table','m'); v.metrics.forEach(m=>{const tr=el('tr'); tr.appendChild(el('td',null,m[0])); tr.appendChild(el('td',null,m[1])); tr.appendChild(el('td',m[2],m[2])); t.appendChild(tr);}); d.appendChild(t);}
  if(v.score_steps?.length){const box=el('div','calc'); box.appendChild(el('b',null,'How this turn was scored')); const list=el('ol'); v.score_steps.forEach(step=>list.appendChild(el('li',null,step))); box.appendChild(list); d.appendChild(box);}
  v.notes.forEach(n=>d.appendChild(el('div','blk',n)));
  return d;
}
function annotation(r){
  const a=A[r.key]=A[r.key]||{}; const box=el('div','ann');
  const lab=el('select'); [['','-- Correct/Incorrect --'],['correct','Correct'],['incorrect','Incorrect']].forEach(o=>{const op=el('option',null,o[1]); op.value=o[0]; lab.appendChild(op);}); lab.value=a.human_label||'';
  const ft=el('select'); const o0=el('option',null,'-- failure type --'); o0.value=''; ft.appendChild(o0);
  D.failure_types.concat(D.extra_types).forEach(t=>{const op=el('option',null,t); op.value=t; ft.appendChild(op);}); ft.value=a.failure_type||'';
  const nt=el('textarea'); nt.placeholder='notes'; nt.value=a.notes||'';
  lab.onchange=()=>{a.human_label=lab.value; save();}; ft.onchange=()=>{a.failure_type=ft.value; save();}; nt.oninput=()=>{a.notes=nt.value; save();};
  box.appendChild(lab); box.appendChild(document.createTextNode(' ')); box.appendChild(ft);
  if(['low','mid','high'].includes(a.human_score_band)) box.appendChild(el('span','badge','Earlier three-level rating: '+a.human_score_band));
  box.appendChild(nt); return box;
}
function turnCard(r){
    const anyFail=r.judges.mtrag.label==='incorrect';
    if($('fTodo').checked && done(r)) return null;
    if($('fFail').checked && !anyFail) return null;
    const c=el('details','card'); c.dataset.key=r.key; c.open=r.key===D.rows[0]?.key;
    const hd=el('summary','hd'); hd.appendChild(el('b',null,'Turn '+r.turn+'/'+r.n_turns));
    hd.appendChild(el('span','badge','answerability: '+r.answerability));
    const jb=el('span','judges'); Object.keys(r.judges).forEach(n=>{jb.appendChild(badge(n,r.judges[n])); jb.appendChild(document.createTextNode(' '));}); hd.appendChild(jb);
    c.appendChild(hd);
    c.appendChild(block('Question', r.question)); c.appendChild(block('RAG answer', r.answer)); c.appendChild(block('Gold answer', r.gold));
    const cx=el('details'); cx.appendChild(el('summary',null,r.contexts===null?'Selected context: not recorded':'Selected context ('+r.contexts.length+' passages)'));
    (r.contexts||[]).forEach((p,i)=>cx.appendChild(block('['+(i+1)+']', p))); c.appendChild(cx);
    const jd=el('div','judges'); Object.keys(r.judges).forEach(n=>jd.appendChild(judgeView(n,r.judges[n]))); c.appendChild(jd);
    c.appendChild(annotation(r)); return c;
}
function render(){
  const root=$('cards'); root.textContent='';
    const groups=new Map();
    D.rows.forEach(r=>{if(!groups.has(r.id))groups.set(r.id,[]);groups.get(r.id).push(r);});
    for(const [id,rows] of groups){
        const group=el('details','case'); group.open=id===D.rows[0]?.id;
        const summary=el('summary'); summary.appendChild(el('span',null,'ID: '+id));
        summary.appendChild(el('span','badge',rows.length+' turns'));
        const judged=rows.filter(r=>r.judges.mtrag.status==='judged').length;
        summary.appendChild(el('span','badge','mtRAG: '+judged+'/'+rows.length+' turns'));
        const turns=el('div','case-turns'); rows.forEach(r=>{const card=turnCard(r);if(card)turns.appendChild(card);});
        group.appendChild(summary);group.appendChild(turns);root.appendChild(group);
    }
  progress();
}
function rowsOut(){
  return D.rows.filter(r=>A[r.key] && (A[r.key].human_score_band||A[r.key].human_label||A[r.key].failure_type||A[r.key].notes)).map(r=>({
    id:r.id, turn:r.turn, human_score_band:A[r.key].human_score_band||'',
    human_label:A[r.key].human_label||'', failure_type:A[r.key].failure_type||'',
    notes:A[r.key].notes||'', annotator:$('annotator').value}));
}
function download(name, text, type){const a=el('a'); a.href=URL.createObjectURL(new Blob([text],{type:type})); a.download=name; a.click();}
$('expJson').onclick=()=>download('annotations_'+D.run+'.json', JSON.stringify(rowsOut(),null,2), 'application/json');
$('expCsv').onclick=()=>{
  const q=s=>'"'+String(s).replace(/"/g,'""')+'"'; const cols=['id','turn','human_score_band','human_label','failure_type','notes','annotator'];
  download('annotations_'+D.run+'.csv', [cols.join(',')].concat(rowsOut().map(o=>cols.map(c=>q(o[c])).join(','))).join('\n'), 'text/csv');};
$('imp').onchange=ev=>{const f=ev.target.files[0]; if(!f)return; f.text().then(t=>{
  JSON.parse(t).forEach(o=>{const a={human_score_band:o.human_score_band||'',human_label:o.human_label||'',failure_type:o.failure_type,notes:o.notes}; migrateBinaryLabel(a); A[o.id+'#'+o.turn]=a;}); save(); render();});};
['fTodo','fFail'].forEach(i=>$(i).onchange=render);
$('blind').onchange=()=>document.body.classList.toggle('blind', $('blind').checked);
$('annotator').value=localStorage.getItem(KEY+':who')||''; $('annotator').oninput=()=>localStorage.setItem(KEY+':who',$('annotator').value);
$('title').textContent='Run: '+D.run; $('crit').innerHTML=D.criteria_html;
Object.keys(D.judge_titles).forEach(name=>{
  const c=D.coverage[name];
  const state=!c.source?'not run':c.records===D.total_records?'all IDs saved':'partial';
  const label=D.judge_titles[name]+': '+c.records+'/'+D.total_records+' IDs, '+c.turns+'/'+D.rows.length+' turns ('+state+')'+(c.source?' - '+c.source:'');
  $('coverage').appendChild(el('span','badge '+(state==='all IDs saved'?'correct':state==='not run'?'not_run':'needs_review'),label));
});
render();
</script></body></html>
"""


def render_page(data):
    payload = json.dumps(data, ensure_ascii=False)
    # keep the JSON from ever closing the script element or opening an HTML comment
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return PAGE.replace("__DATA__", payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", help="default: <work-dir>/annotate.html")
    args = parser.parse_args(argv)
    work_dir = pathlib.Path(args.work_dir)
    output = pathlib.Path(args.output) if args.output else work_dir / "annotate.html"
    if output.exists():
        raise SystemExit("%s already exists; refusing to overwrite." % output)
    data = build_data(work_dir)
    output.write_text(render_page(data), encoding="utf-8")
    print("wrote %s (%d turns)" % (output, len(data["rows"])))


if __name__ == "__main__":
    main()
