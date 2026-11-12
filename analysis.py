#!/usr/bin/env python3
"""
Score Zoom Scribe transcripts against human (Brightcove) captions.

  .venv/bin/python analysis.py            # analyze every tag found under output/batch-medical/
  .venv/bin/python analysis.py --run medical

Writes output/batch-<run>/analysis.json, which the dashboard reads.
"""
import argparse, collections, glob, json, os, re
from rapidfuzz.distance import Levenshtein

ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SYN = {"mg": "milligram", "mgs": "milligrams", "ml": "milliliter", "mls": "milliliters", "kg": "kilogram",
       "mcg": "microgram", "vs": "versus", "dr": "doctor", "ok": "okay", "ii": "2", "iii": "3",
       "gonna": "going to", "wanna": "want to", "gotta": "got to"}
FILLERS = {"um", "uh", "uhh", "umm", "hmm", "mm", "mhm", "er", "erm", "ah"}
TOKEN_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+|[a-z]+(?:'[a-z]+)*")
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that these
those i you he she we they them his her our your their me my so not no do does did done have has had will would can could
should may might must than then there here what which who whom when where why how all any each both more most other some
such only own same too very just also into over under again about up down out off between through during before after
above below s t don doesn didn isn aren wasn weren won wouldn couldn shouldn""".split())


def _dictionary():
    words = set()
    for p in ("/usr/share/dict/words", "/usr/share/dict/web2"):
        if os.path.exists(p):
            words |= {w.strip().lower() for w in open(p, errors="ignore")}
    return words


DICT = _dictionary()


def in_dictionary(w):
    if w in DICT or w in STOP:
        return True
    cands = [w[:-1] if w.endswith("s") else "", w[:-2] if w.endswith("es") else "", w[:-2] if w.endswith("ed") else "",
             w[:-1] if w.endswith("ed") else "", w[:-3] if w.endswith("ing") else "",
             (w[:-3] + "e") if w.endswith("ing") else "", w[:-2] if w.endswith("ly") else "",
             w[:-2] if w.endswith("'s") else "", w[:-3] + "y" if w.endswith("ies") else "",
             w[:-1] if w.endswith("d") else "", w[:-3] if w.endswith("ier") else "", w[:-2] if w.endswith("er") else ""]
    return any(c and c in DICT for c in cands)


def is_technical(w):
    return len(w) >= 4 and not any(c.isdigit() for c in w) and not in_dictionary(w)


def to_tokens(text):
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace("%", " percent ").replace("&", " and ")
    text = re.sub(r"[-–—/]", " ", text)
    out = []
    for t in TOKEN_RE.findall(text):
        if t[0].isdigit() or t[0] == ".":
            t = t.replace(",", "")
            if t.startswith("."):
                t = "0" + t
        for piece in SYN.get(t, t).split():
            if piece not in FILLERS:
                out.append(piece)
    return out


def words_to_digits(items):
    """items: [(token, time)]. Spoken numbers -> digits so '3.8' and 'three point eight' match."""
    toks = [t for t, _ in items]
    n, i, out = len(toks), 0, []

    def small(i):
        if i < n and toks[i] in TENS:
            v, i = TENS[toks[i]], i + 1
            if i < n and toks[i] in ONES and 1 <= ONES[toks[i]] <= 9:
                v, i = v + ONES[toks[i]], i + 1
            return v, i
        if i < n and toks[i] in ONES:
            return ONES[toks[i]], i + 1
        return None

    while i < n:
        r = small(i)
        if r is None:
            out.append(items[i])
            i += 1
            continue
        v, j = r
        if j < n and toks[j] == "hundred" and v < 10:
            v, j = v * 100, j + 1
            k = j + 1 if j < n and toks[j] == "and" else j
            r2 = small(k)
            if r2 and r2[0] < 100:
                v, j = v + r2[0], r2[1]
        s = str(v)
        if j + 1 < n and toks[j] == "point" and toks[j + 1] in ONES and ONES[toks[j + 1]] <= 9:
            d, j = "", j + 1
            while j < n and toks[j] in ONES and ONES[toks[j]] <= 9:
                d, j = d + str(ONES[toks[j]]), j + 1
            s += "." + d
        out.append((s, items[i][1]))
        i = j
    return out


def parse_ts(s):
    parts = [float(x) for x in s.replace(",", ".").split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def read_vtt(path):
    cues, block = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.strip():
            block.append(line.strip())
            continue
        if block:
            cues.append(block)
        block = []
    if block:
        cues.append(block)
    out = []
    for b in cues:
        ts = next((l for l in b if "-->" in l), None)
        if not ts:
            continue
        a, c = [x.strip().split()[0] for x in ts.split("-->")]
        text = " ".join(l for l in b[b.index(ts) + 1:])
        out.append((parse_ts(a), parse_ts(c), text))
    return out


def human_tokens(vtt):
    items, surface = [], collections.defaultdict(collections.Counter)
    for start, end, text in read_vtt(vtt):
        for w in re.findall(r"[A-Za-z][A-Za-z']*", re.sub(r"<[^>]+>", " ", text)):
            surface[w.lower()][w] += 1
        toks = to_tokens(text)
        for k, t in enumerate(toks):
            items.append((t, start + (end - start) * (k / max(len(toks), 1))))
    return words_to_digits(items), surface


def scribe_tokens(segments):
    items = []
    for s in segments:
        words = s.get("words") or [{"word": s["text"], "start": s["start"]}]
        for w in words:
            for t in to_tokens(w["word"]):
                items.append((t, w["start"]))
    return words_to_digits(items)


def align(ref, hyp):
    """Return error spans [(ref_lo, ref_hi, hyp_lo, hyp_hi)] and the equal word pairs."""
    r, h = [t for t, _ in ref], [t for t, _ in hyp]
    spans, cur = [], None
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(r, h).as_list():
        if tag == "equal":
            if cur:
                spans.append(tuple(cur))
                cur = None
            continue
        if cur:
            cur[1], cur[3] = i2, j2
        else:
            cur = [i1, i2, j1, j2]
    if cur:
        spans.append(tuple(cur))
    # spacing-only differences (post operative / postoperative, etc etc / etc) are formatting, not errors
    spans = [sp for sp in spans if "".join(r[sp[0]:sp[1]]) != "".join(h[sp[2]:sp[3]])]
    return r, h, spans


def analyze_video(name, vtt, segments_path):
    ref, surface = human_tokens(vtt)
    hyp = scribe_tokens(json.load(open(segments_path)))
    r, h, spans = align(ref, hyp)
    S = D = I = 0
    rows, missed_idx, phrase_errs, variants = [], set(), collections.Counter(), collections.defaultdict(collections.Counter)
    miss_count = collections.Counter()
    minute_err, minute_n = collections.Counter(), collections.Counter()
    for _, t in ref:
        minute_n[int(t // 60)] += 1
    pos = 0
    for lo, hi, hlo, hhi in spans:
        la, lb = hi - lo, hhi - hlo
        S += min(la, lb)
        D += max(0, la - lb)
        I += max(0, lb - la)
        t = ref[min(lo, len(ref) - 1)][1]
        minute_err[int(t // 60)] += max(la, lb)
        if lo > pos:
            rows.append(["=", " ".join(r[pos:lo]), " ".join(r[pos:lo]), round(ref[pos][1], 1)])
        rows.append(["x", " ".join(r[lo:hi]), " ".join(h[hlo:hhi]), round(t, 1)])
        pos = hi
        heard = " ".join(h[hlo:hhi]) or "(dropped)"
        for k in range(lo, hi):
            missed_idx.add(k)
            w = r[k]
            miss_count[w] += 1
            variants[w][h[hlo + (k - lo)] if la == lb else heard] += 1
        if la >= 2 or lb >= 2:
            phrase_errs[(" ".join(r[lo:hi]), heard)] += 1
    if pos < len(r):
        rows.append(["=", " ".join(r[pos:]), " ".join(r[pos:]), round(ref[pos][1], 1)])
    N = len(r)
    ref_counts = collections.Counter(r)
    tech_idx = [k for k, w in enumerate(r) if is_technical(w)]
    minutes = [{"minute": m, "words": minute_n[m], "errors": minute_err[m],
                "accuracy": round(max(0.0, 1 - minute_err[m] / minute_n[m]), 4)} for m in sorted(minute_n) if minute_n[m] >= 20]
    return {
        "name": name,
        "ref_words": N, "hyp_words": len(h),
        "subs": S, "dels": D, "ins": I,
        "wer": round((S + D + I) / N, 4), "accuracy": round(max(0.0, 1 - (S + D + I) / N), 4),
        "tech_words": len(tech_idx), "tech_missed": sum(1 for k in tech_idx if k in missed_idx),
        "tech_accuracy": round(1 - sum(1 for k in tech_idx if k in missed_idx) / max(len(tech_idx), 1), 4),
        "duration_sec": round(ref[-1][1], 1) if ref else 0,
        "minutes": minutes, "alignment": rows,
        "ref_counts": dict(ref_counts), "miss_counts": dict(miss_count),
        "variants": {w: dict(c.most_common(6)) for w, c in variants.items()},
        "phrase_errors": [[a, b, c] for (a, b), c in phrase_errs.most_common(300)],
        "surface": {w: c.most_common(1)[0][0] for w, c in surface.items()},
    }


def merge_missed(videos):
    ref, miss, var = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    surface = collections.defaultdict(collections.Counter)
    for v in videos:
        ref.update(v["ref_counts"])
        miss.update(v["miss_counts"])
        for w, c in v["variants"].items():
            var[w].update(c)
        for w, s in v["surface"].items():
            surface[w][s] += 1
    rows = []
    for w, m in miss.items():
        rows.append({"word": w, "display": surface[w].most_common(1)[0][0] if surface[w] else w,
                     "occurrences": ref[w], "missed": m, "miss_rate": round(m / ref[w], 3),
                     "technical": is_technical(w), "stopword": w in STOP,
                     "heard_as": dict(var[w].most_common(5)),
                     "videos": sum(1 for v in videos if w in v["miss_counts"])})
    rows.sort(key=lambda x: (-x["missed"], -x["miss_rate"]))
    phrases = collections.Counter()
    for v in videos:
        for a, b, c in v["phrase_errors"]:
            phrases[(a, b)] += c
    return rows, [[a, b, c] for (a, b), c in phrases.most_common(200)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="medical")
    ap.add_argument("--vtt-dir", default="input/brightcove_vtt")
    a = ap.parse_args()
    base = f"output/batch-{a.run}"
    result = {"reference": "Brightcove WebVTT captions (assumed human-produced)", "tags": {}}
    for d in sorted(glob.glob(f"{base}/scribe-*/")):
        tag = os.path.basename(d.rstrip("/")).replace("scribe-", "")
        vids = []
        for vtt in sorted(glob.glob(f"{a.vtt_dir}/*.vtt")):
            name = os.path.basename(vtt).split(".")[0]
            seg = f"{d}{name.lower().replace('_', '-')}.segments.json"
            if os.path.exists(seg):
                v = analyze_video(name, vtt, seg)
                raw = glob.glob(f"{d}raw/{name.lower().replace('_', '-')}*.json")
                if raw:
                    j = json.load(open(raw[0]))
                    v["scribe_seconds"] = j.get("duration_sec")
                    v["speakers"] = len({s.get("speaker") for s in j["result"]["segments"]})
                vids.append(v)
        if not vids:
            continue
        missed, phrases = merge_missed(vids)
        tot = {k: sum(v[k] for v in vids) for k in ("ref_words", "hyp_words", "subs", "dels", "ins", "tech_words", "tech_missed")}
        errs = tot["subs"] + tot["dels"] + tot["ins"]
        tot.update(wer=round(errs / tot["ref_words"], 4), accuracy=round(1 - errs / tot["ref_words"], 4),
                   tech_accuracy=round(1 - tot["tech_missed"] / max(tot["tech_words"], 1), 4),
                   scribe_seconds=round(sum(v.get("scribe_seconds") or 0 for v in vids), 1))
        for v in vids:
            for k in ("ref_counts", "miss_counts", "variants", "phrase_errors", "surface"):
                v.pop(k)
        result["tags"][tag] = {"videos": vids, "totals": tot, "missed": missed, "phrases": phrases}
        print(f"[{tag}] {len(vids)} videos  accuracy {tot['accuracy']:.2%}  WER {tot['wer']:.2%}  "
              f"technical-term accuracy {tot['tech_accuracy']:.2%}")
        for v in vids:
            print(f"   {v['name']}: acc {v['accuracy']:.2%}  (S{v['subs']} D{v['dels']} I{v['ins']} of {v['ref_words']})  tech {v['tech_accuracy']:.2%}")
    if os.path.exists("glossary_terms_curated.txt"):
        result["glossary_terms"] = [l.strip() for l in open("glossary_terms_curated.txt") if l.strip() and not l.startswith("#")]
    json.dump(result, open(f"{base}/analysis.json", "w"))
    print(f"wrote {base}/analysis.json")


if __name__ == "__main__":
    main()
