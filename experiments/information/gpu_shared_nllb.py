"""The message as what two translations share, measured with a pretrained translation model (NLLB-200). GPU.

For each pair (source S, target T) and each held-out verse, NLLB gives the bits of the target verse in three
conditions:
  H(T|S)      the real source verse
  H(T|S')     a different source verse (a shuffled verse from the same test set): the target language, the style
              and the genre are all still known, only this verse's content is missing
  H(T|-)      an empty source: the decoder alone, a neural language model of the target language
The shared message is I = H(T|S') - H(T|S): the bits the real source verse saves. Shared fraction = I / H(T|S').
H(T|-) is also the strongest estimate so far of each language's information per verse (does the 7% spread survive
a neural model?).

Test set: every tenth chapter of the verses shared by the 56 full Bibles (the same as info_shared.py). Pairs:
English <-> every language NLLB supports, plus all ordered pairs among twelve languages.

    python experiments/information/gpu_shared_nllb.py --bibles data/bible --out runs/info_nllb
Options: --model facebook/nllb-200-distilled-600M (default) or facebook/nllb-200-1.3B; --pairs english|matrix|both.
Resumable: finished pairs are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from atlas.loaders.bible import load_bible  # noqa: E402

FULL = ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Bulgarian', 'Burmese', 'Cebuano', 'Chinese', 'Croatian', 'Czech',
        'Danish', 'Dutch', 'English-WEB', 'English', 'Esperanto', 'Farsi', 'Finnish', 'French', 'German', 'Greek',
        'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Kannada', 'Korean', 'Latin',
        'Lithuanian', 'Malagasy', 'Malayalam', 'Maori', 'Marathi', 'Nepali', 'Norwegian', 'Paite', 'Polish',
        'Portuguese', 'Romanian', 'Russian', 'Serbian', 'Shona', 'Slovak', 'Slovene', 'Somali', 'Spanish', 'Swedish',
        'Tagalog', 'Telugu', 'Thai', 'Turkish', 'Vietnamese', 'Xhosa', 'Zarma']
NLLB = {'Afrikaans': 'afr_Latn', 'Albanian': 'als_Latn', 'Amharic': 'amh_Ethi', 'Arabic': 'arb_Arab',
        'Bulgarian': 'bul_Cyrl', 'Burmese': 'mya_Mymr', 'Cebuano': 'ceb_Latn', 'Chinese': 'zho_Hant',
        'Croatian': 'hrv_Latn', 'Czech': 'ces_Latn', 'Danish': 'dan_Latn', 'Dutch': 'nld_Latn',
        'English-WEB': 'eng_Latn', 'English': 'eng_Latn', 'Esperanto': 'epo_Latn', 'Farsi': 'pes_Arab',
        'Finnish': 'fin_Latn', 'French': 'fra_Latn', 'German': 'deu_Latn', 'Greek': 'ell_Grek', 'Hebrew': 'heb_Hebr',
        'Hindi': 'hin_Deva', 'Hungarian': 'hun_Latn', 'Icelandic': 'isl_Latn', 'Indonesian': 'ind_Latn',
        'Italian': 'ita_Latn', 'Japanese': 'jpn_Jpan', 'Kannada': 'kan_Knda', 'Korean': 'kor_Hang',
        'Lithuanian': 'lit_Latn', 'Malagasy': 'plt_Latn', 'Malayalam': 'mal_Mlym', 'Maori': 'mri_Latn',
        'Marathi': 'mar_Deva', 'Nepali': 'npi_Deva', 'Norwegian': 'nob_Latn', 'Polish': 'pol_Latn',
        'Portuguese': 'por_Latn', 'Romanian': 'ron_Latn', 'Russian': 'rus_Cyrl', 'Serbian': 'srp_Cyrl',
        'Shona': 'sna_Latn', 'Slovak': 'slk_Latn', 'Slovene': 'slv_Latn', 'Somali': 'som_Latn', 'Spanish': 'spa_Latn',
        'Swedish': 'swe_Latn', 'Tagalog': 'tgl_Latn', 'Telugu': 'tel_Telu', 'Thai': 'tha_Thai', 'Turkish': 'tur_Latn',
        'Vietnamese': 'vie_Latn', 'Xhosa': 'xho_Latn'}  # no Latin, Paite or Zarma in NLLB
MATRIX = ["English", "German", "French", "Spanish", "Russian", "Finnish", "Hungarian", "Turkish", "Arabic", "Hindi",
          "Chinese", "Japanese"]
SR = {"lj": "љ", "nj": "њ", "dž": "џ", "Lj": "Љ", "Nj": "Њ", "Dž": "Џ", "LJ": "Љ", "NJ": "Њ", "DŽ": "Џ"}
SR1 = dict(zip("abvgdđežzijklmnoprstćufhcčšABVGDĐEŽZIJKLMNOPRSTĆUFHCČŠ",
               "абвгдђежзијклмнопрстћуфхцчшАБВГДЂЕЖЗИЈКЛМНОПРСТЋУФХЦЧШ"))


def to_cyrillic(t: str) -> str:
    for k, v in SR.items():
        t = t.replace(k, v)
    return "".join(SR1.get(c, c) for c in t)


def prepare(name: str, text: str) -> str:
    if name == "Chinese":
        return text.replace(" ", "")
    if name == "Serbian":
        return to_cyrillic(text)
    return text


def bits_batch(model, tok, src_texts, tgt_texts, src_code, tgt_code, device, torch):
    """Bits of each target text given its source text (sum of -log2 p over target tokens after the language tag)."""
    tok.src_lang, tok.tgt_lang = src_code, tgt_code
    enc = tok(src_texts, text_target=tgt_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    labels = enc.pop("labels")
    with torch.no_grad():
        logits = model(**enc, labels=labels).logits.float()
    logp = torch.log_softmax(logits, -1)
    pad = tok.pad_token_id
    lp = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (labels != pad) & (labels != -100)
    mask[:, 0] = False  # the target language tag is given, not predicted
    return (-(lp * mask).sum(1) / np.log(2)).cpu().numpy(), mask.sum(1).cpu().numpy()


def score(model, tok, src, tgt, sc, tc, device, torch, bs):
    order = np.argsort([len(t) for t in tgt])
    out = np.zeros(len(tgt))
    ntok = np.zeros(len(tgt))
    for k in range(0, len(order), bs):
        ix = order[k:k + bs]
        b, n = bits_batch(model, tok, [src[i] for i in ix], [tgt[i] for i in ix], sc, tc, device, torch)
        out[ix] = b
        ntok[ix] = n
    return out, ntok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bibles", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--pairs", choices=["english", "matrix", "both"], default="both")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    langs = [n for n in FULL if (a.bibles / f"{n}.xml").exists()]
    if "English" not in langs:
        sys.exit(f"No English.xml in {a.bibles}: point --bibles at the bible-corpus folder")
    bibles = {n: load_bible(a.bibles / f"{n}.xml") for n in langs}
    ref = list(bibles["English"].keys())
    common = [v for v in ref if all(v in bibles[n] for n in langs)]
    chapters = list(dict.fromkeys(".".join(v.split(".")[:2]) for v in common))
    test_ch = set(chapters[::10])
    test = [v for v in common if ".".join(v.split(".")[:2]) in test_ch]
    rng = random.Random(a.seed)
    shuf = list(range(len(test)))
    while any(i == j for i, j in enumerate(shuf)):
        rng.shuffle(shuf)
    nl = [n for n in langs if n in NLLB]
    pairs = []
    if a.pairs in ("english", "both"):
        pairs += [("English", t) for t in nl if t != "English"] + [(s, "English") for s in nl if s != "English"]
    if a.pairs in ("matrix", "both"):
        pairs += [(s, t) for s, t in permutations(MATRIX, 2) if "English" not in (s, t)]
    print(f"{len(langs)} Bibles, {len(common)} shared verses, {len(test)} test verses, {len(pairs)} pairs, "
          f"device {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model, torch_dtype=torch.float16 if device == "cuda" else None)
    model.to(device).eval()
    (a.out / "test_verses.json").write_text(json.dumps({"verses": test, "shuffle": shuf, "model": a.model}))
    texts = {n: [prepare(n, bibles[n][v]) for v in test] for n in langs}
    # decoder alone, once per target language
    lm_file = a.out / "lm.json"
    lm = json.loads(lm_file.read_text()) if lm_file.exists() else {}
    for t in sorted({t for _, t in pairs}):
        if t in lm:
            continue
        b, n = score(model, tok, [""] * len(test), texts[t], NLLB[t], NLLB[t], device, torch, a.batch)
        lm[t] = {"bits": b.tolist(), "tokens": n.tolist()}
        lm_file.write_text(json.dumps(lm))
        print(f"LM {t}: {b.mean():.1f} bits per verse", flush=True)
    for s, t in pairs:
        f = a.out / f"pair_{s}__{t}.json"
        if f.exists():
            continue
        src = texts[s]
        b_real, n = score(model, tok, src, texts[t], NLLB[s], NLLB[t], device, torch, a.batch)
        b_shuf, _ = score(model, tok, [src[j] for j in shuf], texts[t], NLLB[s], NLLB[t], device, torch, a.batch)
        f.write_text(json.dumps({"source": s, "target": t, "real": b_real.tolist(), "shuffled": b_shuf.tolist()}))
        print(f"{s}->{t}: given source {b_real.mean():.1f}, given other verse {b_shuf.mean():.1f}, "
              f"shared {(b_shuf.mean() - b_real.mean()) / b_shuf.mean():.2f}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
