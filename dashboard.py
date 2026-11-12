"""
Zoom Scribe vs. human captions - management dashboard.

  .venv/bin/python analysis.py && .venv/bin/python glossary.py
  .venv/bin/streamlit run dashboard.py
"""
import base64, json, os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import glossary as gl
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

GREEN, AMBER, RED, BLUE = "#2e9e5b", "#e0a030", "#d64545", "#3b6fd4"

st.set_page_config(page_title="Zoom Scribe vs. Human Captions", page_icon="🎙️", layout="wide")


ENC_PATH = "data/analysis.enc"
LOCAL_PATH = os.environ.get("ZOOM_ANALYSIS_JSON", "")


@st.cache_resource(show_spinner="Unlocking...")
def decrypt_data(password: str):
    blob = open(ENC_PATH, "rb").read()
    salt, token = blob[:16], blob[16:]
    key = base64.urlsafe_b64encode(
        Scrypt(salt=salt, length=32, n=2**17, r=8, p=1).derive(password.encode()))
    try:
        return json.loads(Fernet(key).decrypt(token))
    except InvalidToken:
        return None


def load_data():
    if LOCAL_PATH and os.path.exists(LOCAL_PATH):
        return json.load(open(LOCAL_PATH))
    if st.session_state.get("data"):
        return st.session_state["data"]
    st.title("Zoom Scribe vs. Human Captioning")
    st.caption("Enter the access password to view this dashboard.")
    entered = st.text_input("Password", type="password")
    if entered:
        d = decrypt_data(entered)
        if d is None:
            st.error("Incorrect password.")
        else:
            st.session_state["data"] = d
            st.rerun()
    st.stop()


path = ENC_PATH
data = load_data()
tags = list(data["tags"])

RUN_LABELS = {"plain": "No custom vocabulary", "vocab": "Generic medical vocabulary (47 terms)", "glossary": "Targeted glossary (11 terms)"}

with st.sidebar:
    st.header("Settings")
    tag = st.selectbox("Scribe run", tags, index=tags.index("plain") if "plain" in tags else 0,
                       format_func=lambda t: RUN_LABELS.get(t, t))
    hide_stop = st.checkbox("Hide common words in missed-word list", value=False)
    st.divider()
    st.subheader("Cost assumptions")
    credits_used = st.number_input("Credits used for this run", value=0.379 / 2, format="%.3f",
                                   help="Portal showed 0.379 credits total after two Scribe passes over these 4 videos (plus a few small tests), so ~0.19 per pass. Adjust to your actual figure.")
    human_rate = st.number_input("Human captioning $/media-min", value=0.0, step=0.25,
                                 help="Your vendor/internal rate. Leave 0 to hide the dollar comparison.")
    credit_price = st.number_input("Price per Zoom credit ($)", value=0.0, step=0.05,
                                   help="Your contract rate. Leave 0 to hide the dollar comparison.")
    st.caption("Human reference = existing Brightcove captions (assumed human-produced).")

T = data["tags"][tag]
V, tot = T["videos"], T["totals"]
vdf = pd.DataFrame(V)
vdf["label"] = vdf["name"].str.replace("S3_Medical_Literature_", "Video ", regex=False)
vdf["duration_min"] = vdf["scribe_seconds"].fillna(vdf["duration_sec"]) / 60
total_min = tot["scribe_seconds"] / 60


def color(acc):
    return GREEN if acc >= 0.98 else AMBER if acc >= 0.95 else RED


st.title("Zoom Scribe vs. Human Captioning")
st.caption(f"{len(V)} medical-literature videos · {total_min:.0f} minutes of audio · {tot['ref_words']:,} words in the human transcript")

c = st.columns(5)
c[0].metric("Overall accuracy", f"{tot['accuracy']:.2%}", help="1 − WER against the human captions, after normalizing numbers/spacing/fillers")
c[1].metric("Word error rate", f"{tot['wer']:.2%}")
c[2].metric("Technical-term accuracy", f"{tot['tech_accuracy']:.1%}",
            help="Accuracy on words not found in a standard English dictionary (drug names, gene names, jargon)")
c[3].metric("Best / worst video", f"{vdf.accuracy.max():.1%} / {vdf.accuracy.min():.1%}")
c[4].metric("Zoom credits / media-min", f"{credits_used / total_min:.4f}")

_gl = data["tags"].get("glossary")
_pl = data["tags"].get("plain")
_extra = ""
if _gl and _pl:
    _extra = (f" Adding a targeted glossary of {len(data.get('glossary_terms', []))} terms raised technical-term accuracy from "
              f"{_pl['totals']['tech_accuracy']:.0%} to {_gl['totals']['tech_accuracy']:.0%} (see Vocabulary A/B).")
st.info(
    f"Scribe matched the human captions on **{tot['accuracy']:.1%}** of words overall, but only **{tot['tech_accuracy']:.0%}** of "
    f"technical terms in this run. Plain-English narration is already near human level; the remaining errors are mostly drug and brand names."
    + _extra
)

tab_over, tab_vid, tab_miss, tab_time, tab_cues, tab_ab, tab_gloss, tab_method = st.tabs(
    ["Overview", "Per video", "Most-missed words", "Accuracy over time", "Side-by-side errors", "Vocabulary A/B", "Phonetic glossary", "Method & caveats"])

with tab_over:
    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(vdf, x="label", y="accuracy", text=vdf.accuracy.map("{:.2%}".format), title="Accuracy by video",
                     color="accuracy", color_continuous_scale=[RED, AMBER, GREEN], range_color=[0.95, 1])
        fig.update_yaxes(range=[0.94, 1], tickformat=".0%")
        fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, width='stretch')
    with right:
        errs = pd.DataFrame({"type": ["Substitutions (wrong word)", "Deletions (missed word)", "Insertions (extra word)"],
                             "count": [tot["subs"], tot["dels"], tot["ins"]]})
        fig = px.pie(errs, names="type", values="count", hole=0.5, title="What kind of errors?",
                     color_discrete_sequence=[RED, AMBER, BLUE])
        st.plotly_chart(fig, width='stretch')
    both = pd.DataFrame({"video": vdf.label.tolist() * 2,
                         "kind": ["All words"] * len(vdf) + ["Technical terms only"] * len(vdf),
                         "accuracy": vdf.accuracy.tolist() + vdf.tech_accuracy.tolist()})
    fig = px.bar(both, x="video", y="accuracy", color="kind", barmode="group", title="All words vs. technical terms",
                 color_discrete_sequence=[BLUE, RED], text=both.accuracy.map("{:.1%}".format))
    fig.update_yaxes(range=[0.5, 1.02], tickformat=".0%")
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Whole-set statistics")
    s = st.columns(4)
    s[0].metric("Words (human)", f"{tot['ref_words']:,}")
    s[1].metric("Words (Scribe)", f"{tot['hyp_words']:,}")
    s[2].metric("Total errors", f"{tot['subs'] + tot['dels'] + tot['ins']:,}")
    s[3].metric("Errors per 1,000 words", f"{(tot['subs'] + tot['dels'] + tot['ins']) / tot['ref_words'] * 1000:.1f}")
    s = st.columns(4)
    s[0].metric("Technical words", f"{tot['tech_words']:,}")
    s[1].metric("Technical words missed", f"{tot['tech_missed']:,}")
    s[2].metric("Distinct misspelled terms", f"{sum(1 for m in T['missed'] if m['technical']):,}")
    s[3].metric("Audio processed", f"{total_min:.0f} min")
    if human_rate > 0 and credit_price > 0:
        z, h = credits_used * credit_price, human_rate * total_min
        s = st.columns(3)
        s[0].metric("Zoom cost (this set)", f"${z:,.2f}")
        s[1].metric("Human cost (this set)", f"${h:,.2f}")
        s[2].metric("Zoom as % of human", f"{z / h:.1%}" if h else "n/a")

with tab_vid:
    show = vdf[["name", "duration_min", "ref_words", "accuracy", "wer", "subs", "dels", "ins", "tech_words", "tech_missed", "tech_accuracy"]].copy()
    show.columns = ["Video", "Minutes", "Human words", "Accuracy", "WER", "Substitutions", "Deletions", "Insertions",
                    "Technical words", "Technical missed", "Technical accuracy"]
    st.dataframe(show.style.format({"Minutes": "{:.1f}", "Accuracy": "{:.2%}", "WER": "{:.2%}", "Technical accuracy": "{:.1%}"}),
                 hide_index=True, width='stretch')
    st.caption("Accuracy = 1 − (substitutions + deletions + insertions) / words in the human transcript.")

with tab_miss:
    mdf = pd.DataFrame(T["missed"])
    mdf = mdf[~mdf.stopword] if hide_stop else mdf
    only_tech = st.toggle("Technical / out-of-dictionary terms only", value=True)
    if only_tech:
        mdf = mdf[mdf.technical]
    n = st.slider("How many words", 5, min(60, max(len(mdf), 6)), min(20, max(len(mdf), 5)))
    top = mdf.head(n).copy()
    top["heard"] = top.heard_as.map(lambda d: ", ".join(f"{k} ×{v}" for k, v in d.items()))
    fig = px.bar(top[::-1], x="missed", y="display", orientation="h", color="miss_rate", color_continuous_scale=["#f3c26b", RED],
                 range_color=[0, 1], hover_data={"occurrences": True, "heard": True, "display": False},
                 title="Most commonly missed words (times missed; colour = share of occurrences missed)")
    fig.update_layout(yaxis_title=None, xaxis_title="Times missed", height=max(350, 24 * len(top) + 120))
    st.plotly_chart(fig, width='stretch')
    tbl = top[["display", "occurrences", "missed", "miss_rate", "heard", "videos"]]
    tbl.columns = ["Word (human spelling)", "Times spoken", "Times missed", "Miss rate", "Scribe wrote instead", "Videos affected"]
    st.dataframe(tbl.style.format({"Miss rate": "{:.0%}"}), hide_index=True, width='stretch')
    st.download_button("Download all missed words (CSV)", pd.DataFrame(T["missed"]).drop(columns=["heard_as"]).to_csv(index=False),
                       "missed_words.csv", "text/csv")
    ph = pd.DataFrame(T["phrases"], columns=["Human", "Scribe", "Count"]).head(25)
    ph["Human"] = ph.Human.replace("", "(nothing)")
    with st.expander("Most common multi-word errors"):
        st.dataframe(ph, hide_index=True, width='stretch')

with tab_time:
    rows = [dict(video=v["name"].replace("S3_Medical_Literature_", "Video "), **m) for v in V for m in v["minutes"]]
    mdf2 = pd.DataFrame(rows)
    fig = px.line(mdf2, x="minute", y="accuracy", color="video", markers=True, title="Accuracy per minute of video")
    fig.update_yaxes(tickformat=".0%", range=[max(0.7, mdf2.accuracy.min() - 0.02), 1.005])
    fig.update_layout(xaxis_title="Minute in video", yaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width='stretch')
    worst = mdf2.sort_values("accuracy").head(8)
    st.write("Weakest minutes (usually where new drug names are introduced):")
    st.dataframe(worst.assign(accuracy=worst.accuracy.map("{:.1%}".format)), hide_index=True, width='stretch')

with tab_cues:
    vid = st.selectbox("Video", [v["name"] for v in V])
    v = next(x for x in V if x["name"] == vid)
    errs = [r for r in v["alignment"] if r[0] == "x"]
    st.write(f"{len(errs)} error regions in this video. Human wording on the left, what Scribe wrote on the right.")
    edf = pd.DataFrame(errs, columns=["_", "Human caption", "Scribe", "Time (s)"]).drop(columns="_")
    edf["Time"] = edf["Time (s)"].map(lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}")
    edf["Scribe"] = edf.Scribe.replace("", "(dropped)")
    edf["Human caption"] = edf["Human caption"].replace("", "(extra word)")
    q = st.text_input("Filter")
    if q:
        edf = edf[edf.apply(lambda r: q.lower() in f"{r['Human caption']} {r['Scribe']}".lower(), axis=1)]
    st.dataframe(edf[["Time", "Human caption", "Scribe"]], hide_index=True, width='stretch', height=420)

with tab_ab:
    order = [t for t in ("plain", "vocab", "glossary") if t in data["tags"]]
    st.write("Same four videos, same model, three vocabulary settings.")
    cols = st.columns(len(order))
    base_t = data["tags"]["plain"]["totals"] if "plain" in data["tags"] else None
    for col, t in zip(cols, order):
        tt = data["tags"][t]["totals"]
        col.markdown(f"**{RUN_LABELS.get(t, t)}**")
        col.metric("Word error rate", f"{tt['wer']:.2%}",
                   None if t == "plain" or not base_t else f"{(tt['wer'] - base_t['wer']) * 100:+.2f} pts", delta_color="inverse")
        col.metric("Technical-term accuracy", f"{tt['tech_accuracy']:.1%}",
                   None if t == "plain" or not base_t else f"{(tt['tech_accuracy'] - base_t['tech_accuracy']) * 100:+.1f} pts")
        col.metric("Total errors", f"{tt['subs'] + tt['dels'] + tt['ins']:,}",
                   None if t == "plain" or not base_t else f"{(tt['subs'] + tt['dels'] + tt['ins']) - (base_t['subs'] + base_t['dels'] + base_t['ins']):+,}",
                   delta_color="inverse")
    rows = []
    for t in order:
        for v in data["tags"][t]["videos"]:
            rows.append({"video": v["name"].replace("S3_Medical_Literature_", "Video "), "run": RUN_LABELS.get(t, t),
                         "accuracy": v["accuracy"], "technical accuracy": v["tech_accuracy"]})
    ab = pd.DataFrame(rows)
    l, r_ = st.columns(2)
    for col, key in ((l, "accuracy"), (r_, "technical accuracy")):
        fig = px.bar(ab, x="video", y=key, color="run", barmode="group", title=key.capitalize(),
                     color_discrete_sequence=["#9aa3b2", AMBER, GREEN])
        fig.update_yaxes(tickformat=".1%", range=[0.6 if "tech" in key else 0.95, 1.0])
        fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None, legend=dict(orientation="h", y=-0.2))
        col.plotly_chart(fig, width='stretch')

    if "glossary" in data["tags"] and "plain" in data["tags"] and data.get("glossary_terms"):
        st.subheader("Glossary terms: before and after")
        def _m(t):
            return {m["word"]: m for m in data["tags"][t]["missed"]}
        mp, mg = _m("plain"), _m("glossary")
        refc = {}
        for m in list(mp.values()) + list(mg.values()):
            refc[m["word"]] = m["occurrences"]
        trows = []
        for term in data["glossary_terms"]:
            w = term.lower()
            occ = refc.get(w)
            if occ is None:
                continue
            b_, a_ = mp.get(w, {}).get("missed", 0), mg.get(w, {}).get("missed", 0)
            trows.append({"Term": term, "Times spoken": occ, "Missed without glossary": b_, "Missed with glossary": a_,
                          "Fixed": b_ - a_, "Scribe still wrote": ", ".join(mg.get(w, {}).get("heard_as", {}).keys())})
        tdf = pd.DataFrame(trows).sort_values("Missed without glossary", ascending=False)
        st.dataframe(tdf, hide_index=True, width='stretch')
    st.warning(
        "**Read this as a best case, not a forecast.** The glossary terms were taken from these same four videos, so this shows what "
        "happens when the team knows the drug and brand names in advance (the realistic workflow for course content), not performance "
        "on unseen content. The videos cover different topics and share almost no terms, so a glossary has to be built for each batch "
        "of content. The generic 47-term medical list made no real difference, because it did not contain the names that Scribe actually misses."
    )

with tab_gloss:
    st.write("Terms Scribe got wrong, with a pronunciation guide to give to Zoom (or to a speaker/editor). "
             "Pronunciations are machine-generated from spelling, so **edit the 'say it like' column** for invented names before sending.")
    c1, c2 = st.columns(2)
    min_missed = c1.number_input("Include terms missed at least N times", 1, 50, 1)
    inc_common = c2.toggle("Also include everyday words missed ≥50% of the time", value=False)

    @st.cache_data
    def build_glossary(tag_, min_missed_, inc_, mtime):
        return gl.build(data["tags"][tag_]["missed"], min_missed_, inc_)

    g = pd.DataFrame(build_glossary(tag, int(min_missed), inc_common, 1))
    if g.empty:
        st.write("No terms match.")
    else:
        g.insert(0, "include", True)
        edited = st.data_editor(
            g, hide_index=True, width='stretch', num_rows="fixed",
            column_config={"include": st.column_config.CheckboxColumn("Send to Zoom"),
                           "miss_rate": st.column_config.NumberColumn("miss rate", format="%.2f"),
                           "say_it_like": st.column_config.TextColumn("say it like (editable)"),
                           "ipa": st.column_config.TextColumn("IPA"), "arpabet": st.column_config.TextColumn("ARPAbet")},
            disabled=[c for c in g.columns if c not in ("include", "say_it_like", "term")])
        chosen = edited[edited.include]
        vocab = gl.zoom_vocabulary(chosen.to_dict("records"))
        d1, d2, d3 = st.columns(3)
        d1.download_button("Glossary CSV", chosen.drop(columns="include").to_csv(index=False), "phonetic_glossary.csv", "text/csv")
        d2.download_button("Zoom vocabulary JSON", json.dumps(vocab, indent=2), "zoom_vocabulary.json", "application/json")
        d3.download_button("Plain term list", "\n".join(vocab["vocabulary"]["phrases"]), "terms.txt", "text/plain")
        st.code(json.dumps(vocab, indent=2), language="json")
        st.caption("The JSON goes into a Scribe batch job as `config.vocabulary` (phrases). Zoom's public docs list no per-term "
                   "pronunciation field, so the phonetic column is for human reviewers and prompt-writing.")

with tab_method:
    st.markdown(
        """
**How accuracy is measured**
- Reference: existing Brightcove WebVTT captions for each video. These are treated as the human transcript but were **not independently verified** as human-made or error-free.
- Hypothesis: Zoom Scribe (`zoom-scribe-en`) batch output with word timestamps and diarization.
- Both are normalized the same way: lowercase, punctuation/speaker tags/fillers removed, hyphens split, spoken numbers → digits, `%` → "percent", `mg` → "milligram", spacing-only differences (post operative / postoperative) ignored.
- Accuracy = 1 − WER, with WER from a word-level Levenshtein alignment.
- Technical words = reference words of 4+ letters that are not in a standard English dictionary.

**Caveats**
- Four videos from a single subject area (~115 min) is a small sample.
- Several videos use invented drug/brand names that no speech system has seen; real-world names may do better or worse.
- Some "errors" are the human caption being wrong or stylistically different (e.g. "could" vs. "can"). Scores are against *a* reference, not ground truth.
- A fully independent human pass on a random sample would firm up the comparison.
"""
    )
