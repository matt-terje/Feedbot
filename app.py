import os, json, yaml, textwrap, datetime
import streamlit as st
from typing import Dict, Any, List

# ---- Model client (OpenAI-style). Swap to your preferred provider if needed.
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

st.set_page_config(page_title=CFG["app"]["title"], page_icon="🤖", layout="wide")
st.title(CFG["app"]["title"])
st.caption(CFG["app"]["disclaimer"])

# ---- Sidebar: choose task & view rubric
task_keys = list(CFG["rubrics"].keys())
task_key = st.sidebar.selectbox("Task", task_keys, format_func=lambda k: CFG["rubrics"][k]["label"])
rubric = CFG["rubrics"][task_key]
exemplars = CFG["exemplars"].get(task_key, {})

# Optional: adjust weights on the fly
st.sidebar.subheader("Weights")
weights = {}
total_w = 0
for c in rubric["criteria"]:
    w = st.sidebar.number_input(f'{c["name"]} (out of {c["weight"]})', min_value=0, max_value=20, value=c["weight"])
    weights[c["id"]] = w
    total_w += w
st.sidebar.write(f"**Total weight:** {total_w} (scaled to /20)")

# ---- Main input
st.subheader("Student Submission")
student_name = st.text_input("Student name (optional; use initials if preferred)")
submission = st.text_area("Paste your work here", height=240, placeholder="Paste your journal/portfolio text…")

# Optional parameters
st.subheader("Parameters (optional)")
year_group = st.selectbox("Year group", ["Year 7", "Year 8", "Other"], index=0)
tone = st.selectbox("Feedback tone", ["Supportive & specific", "Exam-prep focused", "Concise bullets"])
next_steps_count = st.slider("Number of next-step suggestions", 2, 6, 3)

# ---- Helper: build system + user prompts
def build_prompt(task_key: str, submission: str, weights: Dict[str, int]) -> List[Dict[str, str]]:
    rub = CFG["rubrics"][task_key]
    ex = CFG["exemplars"].get(task_key, {})
    criteria_bullets = "\n".join([
        f'- {c["name"]} ({weights[c["id"]]}): {c["desc"]}'
        for c in rub["criteria"]
    ])

    # Few-shot exemplars compressed
    examples = []
    for band in ["high", "mid", "low"]:
        if band in ex and ex[band].strip():
            examples.append(f"{band.upper()} exemplar:\n{ex[band].strip()}\n")

    system = f"""You are a teacher's assistant for Stage 4 Technology (Mandatory) in NSW.
You produce fair, explainable feedback aligned to the teacher-provided rubric.
Return a strict JSON object with keys: overall_comment, criteria (list of {{id, score, out_of, feedback}}), next_steps (list), total_score.
- Scores must be non-negative integers.
- Sum of criterion scores scales to /20 based on provided weights.
- Write feedback to the student, not to the teacher.
- No personal data. No grades beyond /20.
- Be supportive, specific, and actionable. Avoid generic praise."""

    user = f"""
TASK: {rub['label']}
YEAR GROUP: {year_group}
TONE: {tone}
WEIGHTS (sum {sum(weights.values())}, scale to /20):
{criteria_bullets}

STUDENT SUBMISSION:
{submission.strip() if submission else "(empty)"}

EXAMPLES (for calibration; do not copy):
{textwrap.dedent("\n".join(examples))}

OUTPUT JSON SCHEMA:
{{
  "overall_comment": "string",
  "criteria": [
    {{"id": "planning", "score": 4, "out_of": 5, "feedback": "string"}},
    ...
  ],
  "next_steps": ["string", "string"],
  "total_score": 17
}}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def call_model(messages: List[Dict[str,str]]) -> Dict[str, Any]:
    # Use a capable, instruction-following model; replace model name to your org standard.
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o" / your preferred
        temperature=0.2,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)

# ---- Run
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Get Feedback", type="primary", disabled=not submission.strip()):
        try:
            msgs = build_prompt(task_key, submission, weights)
            data = call_model(msgs)

            # Safety: validate schema minimally
            assert "criteria" in data and isinstance(data["criteria"], list)
            assert "total_score" in data and isinstance(data["total_score"], (int, float))
            assert "overall_comment" in data
            assert "next_steps" in data and isinstance(data["next_steps"], list)

            st.success(f"Total Score: {int(data['total_score'])}/20")
            st.write("### Overall Comment")
            st.write(data["overall_comment"])

            st.write("### Criteria Feedback")
            for c in data["criteria"]:
                # Find display name
                name = next((r["name"] for r in rubric["criteria"] if r["id"] == c["id"]), c["id"])
                st.markdown(f"**{name}** — {c['score']}/{c['out_of']}")
                st.write(c["feedback"])

            st.write("### Next Steps")
            for i, step in enumerate(data["next_steps"], start=1):
                st.write(f"{i}. {step}")

            # Optional save (anonymised)
            log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "task": task_key,
                "year_group": year_group,
                "weights": weights,
                "submission_len": len(submission),
                "result": data,
            }
            os.makedirs("logs", exist_ok=True)
            with open(f"logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)

        except Exception as e:
            st.error(f"Something went wrong parsing the model output. {e}")

with col2:
    st.write("### Rubric Preview")
    st.caption("These are the criteria and default weights you provided.")
    for c in rubric["criteria"]:
        st.markdown(f"**{c['name']}** (*{c['weight']}*)")
        st.write(c["desc"])
    if exemplars:
        st.write("### Exemplars (teacher-provided)")
        for band in ["high", "mid", "low"]:
            if band in exemplars and exemplars[band].strip():
                st.markdown(f"**{band.title()}**")
                st.code(exemplars[band].strip()[:800] + ("..." if len(exemplars[band]) > 800 else ""))
