"""Native vs translated: a factorial library of novels in English, French and German.

Each work appears in its original language and in translation, so that four things can be pulled apart:
  work      the same story (content), read in any language
  language  the language the text is written in
  origin    the language the work was first written in (culture of origin)
  native    whether this text is the original (language == origin)
Works first written in Russian or Spanish give a control cell where every version is a translation.

Sources: the GITenberg mirror of Project Gutenberg on raw.githubusercontent.com. Multi-volume texts are
concatenated. Each text's language is checked with function-word counts, and a catalogue.csv records work, origin,
language, native flag, words and source repos.

    python experiments/native/fetch_native.py --out data/native
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "texts"))
from fetch import get, strip_gutenberg  # noqa: E402

# work: (origin, {language: [GITenberg repos, in reading order]})
WORKS = {
    # French originals
    "bovary": ("fr", {"fr": ["Madame-Bovary_14155"], "en": ["Madame-Bovary_2413"], "de": ["Frau-Bovary_15711"]}),
    "robur": ("fr", {"fr": ["Robur-le-Conquerant_5126"], "en": ["Robur-the-Conqueror_3808"],
                     "de": ["Robur-der-Sieger_15559"]}),
    "vingtmille": ("fr", {"fr": ["Vingt-mille-Lieues-Sous-Les-Mers---Complete_5097"],
                          "en": ["Twenty-Thousand-Leagues-under-the-Sea_164"]}),
    "tourdumonde": ("fr", {"fr": ["Le-tour-du-monde-en-quatre-vingts-jours_800"],
                           "en": ["Around-the-World-in-80-Days_103"]}),
    "mousquetaires": ("fr", {"fr": ["Les-trois-mousquetaires_13951"], "en": ["The-Three-Musketeers_1257"]}),
    "notredame": ("fr", {"fr": ["Notre-Dame-de-Paris_19657"], "en": ["Notre-Dame-De-Paris_2610"]}),
    "germinal": ("fr", {"fr": ["Germinal_5711"], "en": ["Germinal_56528"]}),
    "ilemyst": ("fr", {"fr": ["L--le-myst-rieuse_14287"], "en": ["The-Mysterious-Island_1268"]}),
    "tulipe": ("fr", {"fr": ["La-tulipe-noire_26504"], "en": ["The-Black-Tulip_965"]}),
    # German originals
    "werther": ("de", {"de": ["Die-Leiden-des-jungen-Werther---Band-1_2407", "Die-Leiden-des-jungen-Werther---Band-2_2408"],
                       "en": ["The-Sorrows-of-Young-Werther_2527"], "fr": ["Werther_62368"]}),
    "zarathustra": ("de", {"de": ["Also-sprach-Zarathustra--Ein-Buch-f-r-Alle-und-Keinen_7205"],
                           "en": ["Thus-Spake-Zarathustra--A-Book-for-All-and-None_1998"],
                           "fr": ["Ainsi-Parlait-Zarathoustra_5258"]}),
    "heidi": ("de", {"de": ["Heidis-Lehr-und-Wanderjahre_7500", "Heidi-kann-brauchen-was-es-gelernt-hat_7512"],
                     "en": ["Heidi_1448"]}),
    "siddhartha": ("de", {"de": ["Siddhartha--eine-indische-Dichtung_2499"], "en": ["Siddhartha_2500"]}),
    "effi": ("de", {"de": ["Effi-Briest_5323"]}),
    "wahlverw": ("de", {"de": ["Die-Wahlverwandtschaften_2403"]}),
    "buddenbrooks": ("de", {"de": ["BuddenbrooksVerfall-einer-Familie_34811"]}),
    "meister": ("de", {"de": [f"Wilhelm-Meisters-Lehrjahre-Band-{i}_{2334 + i}" for i in range(1, 9)]}),
    "verwandlung": ("de", {"de": ["Die-Verwandlung_22367"], "en": ["Metamorphosis_5200"]}),
    # English originals
    "sawyer": ("en", {"en": ["The-Adventures-of-Tom-Sawyer_74"], "de": ["Die-Abenteuer-Tom-Sawyers_30165"]}),  # the "French" file is Catalan
    "alice": ("en", {"en": ["Alice-s-Adventures-in-Wonderland_11"], "de": ["Alice-s-Abenteuer-im-Wunderland_19778"],
                     "fr": ["Aventures-d-Alice-au-pays-des-merveilles_55456"]}),
    "uncletom": ("en", {"en": ["Uncle-Tom-s-Cabin_203"], "fr": ["La-case-de-l-oncle-Tom--13-ou-vie-des-n-gres-en-Am-rique_38704"]}),  # German file is volume 3 only
    "ivanhoe": ("en", {"en": ["Ivanhoe--A-Romance_82"],
                       "fr": ["Ivanhoe--1-4-Le-retour-du-crois-_33315", "Ivanhoe--2-4-Le-retour-du-crois-_34332",
                              "Ivanhoe--3-4-Le-retour-du-crois-_34342", "Ivanhoe--4-4-Le-retour-du-crois-_34608"]}),  # the other two are Dutch and Finnish
    "copperfield": ("en", {"en": ["David-Copperfield_766"],
                           "fr": ["David-Copperfield---Tome-I_17868", "David-Copperfield---Tome-II_17869"]}),
    "gulliver": ("en", {"en": ["Gulliver-s-Travels_829"], "fr": ["Les-Voyages-de-Gulliver_17640"]}),
    "crusoe": ("en", {"en": ["The-Life-and-Adventures-of-Robinson-Crusoe_521"],
                      "fr": ["Aventures-surprenantes-de-Robinson-Crusoe_57964"]}),
    "treasure": ("en", {"en": ["Treasure-Island_120"], "de": ["Die-Schatzinsel_49424"]}),
    "oliver": ("en", {"en": ["Oliver-Twist_730"], "fr": ["Oliver-Twist_16023"], "de": ["Oliver-Twist_56586"]}),
    # control: neither language is the original
    "karenina": ("ru", {"en": ["Anna-Karenina_1399"], "de": ["Anna-Karenina-1.-Band_44956", "Anna-Karenina-2.-Band_44957"]}),
    "warpeace": ("ru", {"en": ["War-and-Peace_2600"],
                        "fr": ["La-guerre-et-la-paix-Tome-I_17949", "La-guerre-et-la-paix-Tome-II_17950",
                               "La-guerre-et-la-paix-Tome-III_17951"]}),
    "quixote": ("es", {"en": ["Don-Quixote_996"],
                       "fr": ["L-ing-nieux-hidalgo-Don-Quichotte-de-la-Manche---Tome-I_16066",
                              "L-ing-nieux-hidalgo-Don-Quichotte-de-la-Manche---Tome-II_16067"]}),
}

FUNC = {"en": {"the", "and", "of", "to", "was", "he", "that", "it", "with", "his"},
        "fr": {"le", "la", "les", "et", "de", "il", "une", "que", "est", "dans"},
        "de": {"der", "die", "und", "das", "nicht", "ist", "ich", "zu", "den", "sie"}}


def detect(text: str) -> tuple[str, float]:
    words = re.findall(r"\w+", text[:400_000].lower())
    share = {k: sum(w in v for w in words) / max(len(words), 1) for k, v in FUNC.items()}
    best = max(share, key=share.get)
    return best, share[best]


def fetch_repo(repo: str) -> str | None:
    num = repo.rsplit("_", 1)[1]
    for fn in (f"{num}-0.txt", f"{num}.txt", f"{num}-8.txt"):
        try:
            raw = get(f"https://raw.githubusercontent.com/GITenberg/{repo}/master/{fn}")
        except Exception:
            continue
        try:
            t = raw.decode("utf-8")
        except UnicodeDecodeError:
            t = raw.decode("latin-1")
        return strip_gutenberg(t.replace("\r", ""))
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*", help="fetch only these works (the catalogue is then appended to)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for work, (origin, versions) in WORKS.items():
        if a.only and work not in a.only:
            continue
        for slot, repos in versions.items():
            parts = []
            for r in repos:
                t = fetch_repo(r)
                if t is None:
                    print(f"{work}/{slot}: {r} not found", file=sys.stderr)
                    parts = None
                    break
                parts.append(t)
            if not parts:
                continue
            text = "\n\n".join(parts)
            lang, share = detect(text)
            if slot in FUNC and lang != slot:
                print(f"{work}/{slot}: detected {lang} ({share:.2f}), kept under detected language", file=sys.stderr)
            name = f"{work}--{lang}"
            (a.out / f"{name}.txt").write_text(text)
            words = len(re.findall(r"\w+", text))
            rows.append({"name": name, "work": work, "origin": origin, "language": lang,
                         "native": int(lang == origin), "words": words, "func_share": round(share, 3),
                         "repos": " ".join(repos)})
            print(f"{name}: {words} words, {lang} {share:.2f}", flush=True)
    cat = a.out / "catalogue.csv"
    if a.only and cat.exists():
        with open(cat, newline="") as fh:
            old = [r for r in csv.DictReader(fh) if r["work"] not in a.only]
        rows = old + rows
    with open(cat, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
