# Native vs translated texts (2026-11-26)

Question: every earlier cross-language result used translations, where the language layer of concept geometry came
out near zero. Does a language layer appear when texts are written natively, and does information survive the step
from an original to a translation?

Code: fetch_native.py (library), analyse_native.py (geometry), info_native.py (information), run_native.sh (GPU).

## Library

- 62 public-domain novels and books from the GITenberg mirror of Project Gutenberg, covering 31 works.
  - French originals (9): Bovary, Robur, Vingt mille lieues, Tour du monde, Trois mousquetaires, Notre-Dame,
    Germinal, Île mystérieuse, Tulipe noire.
  - German originals (9): Werther, Zarathustra, Heidi, Siddhartha, Verwandlung, plus four German-only novels
    (Effi Briest, Wahlverwandtschaften, Buddenbrooks, Wilhelm Meisters Lehrjahre).
  - English originals (9): Tom Sawyer, Alice, Uncle Tom's Cabin, Ivanhoe, Copperfield, Gulliver, Crusoe,
    Treasure Island, Oliver Twist.
  - Control (every version a translation): Anna Karenina, War and Peace (Russian), Don Quixote (Spanish).
- 26 English, 20 French and 16 German texts. Each work appears in 1-3 languages.
- Six mislabelled files were found and dropped (the "French" Tom Sawyer is Catalan; others were Dutch, Finnish and
  Greek). The German Uncle Tom is volume 3 only and was dropped.
- Instruments: counting (svd, ppmi) on the 123 aligned concepts of probes_wide, frequency partialled. 51 concepts
  occur in every long text.
- Split-half reliability is low, as before with counting: median 0.30 (svd), 0.39 (ppmi).

## 1. A language layer does appear, and it is the same in native and translated text

Pair model (1,891 pairs; RSA regressed on pair type, pair reliability as covariate; CIs from resampling works):

| effect | svd | ppmi |
|---|---|---|
| same work (same content) | +0.111 (0.096-0.128) | +0.123 (0.104-0.144) |
| same language | +0.038 (0.030-0.048) | +0.040 (0.031-0.050) |
| same origin (culture of the original) | +0.004 (-0.004-0.012) | +0.005 (-0.002-0.014) |
| same language, both native (extra) | -0.001 (-0.013-0.012) | +0.002 (-0.009-0.017) |

- The language effect is about a third of the content effect.
- Being native adds nothing beyond it. Native French and native English novels differ by the same amount as two
  translations into French and English do.
- The culture of origin leaves no measurable trace in the pair model. A faint "shining through" exists: a
  translation sits slightly closer to native texts of its source language than a native text of its own language
  does (+0.007, CI 0.002-0.012, svd; +0.011, CI 0.006-0.015, ppmi; 62-79% of translations).
- Disattenuated cell values (svd, median): same work original/translation 0.88; native/native same language 0.54;
  native/native other language 0.38; translated/translated same language 0.54-0.56; other language 0.44.

## 2. The language difference is a stable fingerprint, mostly idiom

Difference vectors over concept pairs, each split in half over works:

| | native difference split-half | translation-pair difference split-half | native vs translation difference |
|---|---|---|---|
| English-French | 0.23 | 0.34 | 0.68 |
| German-English | 0.22 | 0.16 | 0.56 |
| German-French | 0.22 | 0.07 | 0.49 |

Null for the native split-half (language labels mixed): 0.00, sd 0.05. (svd; ppmi is the same within 0.05.)

- The difference between native French and native English novels is mostly the same difference found between the
  French and English versions of one work. So it is the language, not culture or story.
- The concept pairs that drive it are collocations and how words divide their senses: English "time" goes with
  first/second ("the first time"; French uses "fois", not "temps"); "question/ask" ("poser une question");
  French "high/voice" ("à haute voix", aloud); "great/long".
- This revises the earlier reading that language is about zero. The earlier test had only two kinds of text
  (Bible, Europarl), and the content difference swamped a small language difference. With 19 works in both
  English and French, the language layer is small but reliable.

## 3. Information survives translation better than length does

Each version is compressed with PPMd (orders 6 and 16). For each original/translation pair, the log ratio of
translation to original minus the Bible's log ratio for the same two languages (4,583 verses, cut to the median novel
length) removes the language. Pairs whose word ratio is more than 15% off the Bible's are flagged as abridged or
padded (vingtmille-en, tourdumonde-en, robur-en, werther, crusoe-fr) and left out.

- 23 complete original/translation pairs: translations carry +1.0% bits (sd 9.8%, 15 of 23 positive) and +2.5%
  characters (sd 11.4%). There's no detectable inflation of information in translation (the CI on the mean is about
  plus or minus 4%).
- Mean absolute gap between an original and its translation: 10.4% in characters, 6.4% in bits.
- So, like the Bible translations, an original and its translation agree more closely in information than in length.

## What this changes

- The layered picture needs a language layer after all: content > language (about 1/3 of content) > origin culture
  (about 0). Language lives in idiom and word senses, and it is there whether a text is native or translated.
- What does not show up in native text is anything extra: no native-only language effect, no cultural signature.
- Information conservation holds from original to translation, at about the same tightness as between Bible
  translations.

## Caveats

- Counting instruments see word forms and collocations, so they are built to see a lexical language layer. The
  neural text effect (run_native.sh) is the check.
- Native works in different languages also differ in content and era. The model separates these only as far as the
  translation pairs allow (19 English-French works, 9 German-English, 5 German-French).
- Old translations are uneven (abridgements, prefaces). The word-ratio filter catches the large cases only.
- Exploratory; not preregistered.
