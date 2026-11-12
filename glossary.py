#!/usr/bin/env python3
"""
Build a phonetic glossary of the terms Scribe most often gets wrong.

  .venv/bin/python glossary.py [--tag plain] [--min-missed 1] [--include-common]

Writes output/batch-medical/glossary.csv, glossary.json and zoom_vocabulary.json
(the last is ready to paste into a Scribe job as  config.vocabulary).
"""
import argparse, csv, json, re

ARPA_RESPELL = {
    "AA": "ah", "AE": "a", "AH": "uh", "AO": "aw", "AW": "ow", "AY": "eye", "EH": "eh", "ER": "er", "EY": "ay",
    "IH": "ih", "IY": "ee", "OW": "oh", "OY": "oy", "UH": "uu", "UW": "oo",
    "B": "b", "CH": "ch", "D": "d", "DH": "th", "F": "f", "G": "g", "HH": "h", "JH": "j", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t", "TH": "th", "V": "v",
    "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}
ARPA_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ",
    "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f",
    "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

_g2p = None


def phonemes(word):
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    return [p for p in _g2p(word) if re.match(r"[A-Z]+\d?$", p)]


def respell(phs):
    """ARPAbet -> respelling like meh-KAN-ih-bih: split at vowels, stressed syllable in CAPS."""
    items = [re.match(r"([A-Z]+)(\d?)", p).groups() for p in phs]
    vow = [i for i, (b, st) in enumerate(items) if st != ""]
    if not vow:
        return "".join(ARPA_RESPELL.get(b, b.lower()) for b, _ in items)
    cuts = [0]
    for v1, v2 in zip(vow, vow[1:]):
        gap = v2 - v1 - 1
        cuts.append(v2 if gap <= 1 else v1 + 2)
    cuts.append(len(items))
    out = []
    for k in range(len(vow)):
        seg = items[cuts[k]:cuts[k + 1]]
        txt = "".join(ARPA_RESPELL.get(b, b.lower()) for b, _ in seg)
        out.append(txt.upper() if any(st == "1" for _, st in seg) else txt)
    return "-".join(out)


def build(missed, min_missed=1, include_common=False, min_rate=0.5):
    rows = []
    for m in missed:
        if m["stopword"] or m["missed"] < min_missed:
            continue
        w = m["word"]
        if not (m["technical"] or include_common) or w.isdigit():
            continue
        if not m["technical"] and m["miss_rate"] < min_rate:
            continue
        phs = phonemes(w)
        rows.append({
            "term": m["display"] if m["display"] else w,
            "times_in_content": m["occurrences"], "times_missed": m["missed"], "miss_rate": m["miss_rate"],
            "heard_as": ", ".join(f"{k} ({v})" for k, v in m["heard_as"].items()),
            "arpabet": " ".join(phs),
            "ipa": "".join(ARPA_IPA.get(re.sub(r"\d", "", p), "") for p in phs),
            "say_it_like": respell(phs),
        })
    rows.sort(key=lambda r: (-r["times_missed"], -r["miss_rate"]))
    return rows


def zoom_vocabulary(rows, boost_phrases=()):
    seen, phrases = set(), []
    for r in rows:
        for t in (r["term"], *boost_phrases):
            if t.lower() not in seen:
                seen.add(t.lower())
                phrases.append(t)
    return {"vocabulary": {"phrases": phrases}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="medical")
    ap.add_argument("--tag", default="plain")
    ap.add_argument("--min-missed", type=int, default=1)
    ap.add_argument("--include-common", action="store_true")
    a = ap.parse_args()
    base = f"output/batch-{a.run}"
    data = json.load(open(f"{base}/analysis.json"))["tags"][a.tag]
    rows = build(data["missed"], a.min_missed, a.include_common)
    with open(f"{base}/glossary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(f"{base}/glossary.json", "w"), indent=2)
    json.dump(zoom_vocabulary(rows), open(f"{base}/zoom_vocabulary.json", "w"), indent=2)
    print(f"{len(rows)} glossary terms -> {base}/glossary.csv")
    for r in rows[:12]:
        print(f"  {r['term']:<16} {r['say_it_like']:<22} {r['ipa']:<18} missed {r['times_missed']}/{r['times_in_content']}")


if __name__ == "__main__":
    main()
