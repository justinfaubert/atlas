# Information rate across Bible translations (2026-11-25)

Question: every Atlas measure so far compared similarity between pairs of words. Here we measure something
different: how many bits it takes to write the same content down in each language. If languages differ in length
but not in information, then information is a conserved quantity and length is just how it gets packaged.

Code: experiments/information/info_rate.py, analyse_info.py. No GPU. Takes about 35 minutes on 4 CPUs.
Data: christos-c Bible corpus.
- Full set: 56 whole Bibles, 24,563 verses that every one of them has.
- NT set: 84 New Testaments, 6,132 verses.

Two instruments:
- lzma: the whole text compressed, with characters recoded by frequency rank so scripts that need several bytes
  per character aren't penalised.
- Character model: order 6, Witten-Bell smoothing. Each book is scored with that book left out of training. This
  gives bits for every verse and for every model order.

The two instruments rank languages alike (Spearman 0.89).

## 1. Information is close to conserved; length is not

Spread across the 56 full Bibles, as the sd of log values in %:

| measure | spread | max / min |
|---|---|---|
| UTF-8 bytes | 37% | 3.7x |
| characters (no spaces) | 27% (CI 17-35) | 4.1x |
| words | 24% | 3.3x |
| lzma bits | 9% (CI 6-11) | 1.7x |
| character-model bits | 7% (CI 5-9) | 1.5x |

- The median is about 230 bits per verse. Chinese uses 0.8M characters and English 2.7M, but they need 212 and
  216 bits per verse.
- The slope of log bits on log characters is 0.11 (CI 0.05-0.19). Here 0 would mean information has nothing to do
  with length, and 1 would mean information grows in proportion to length.
- Translation floor: English (KJV) and English-WEB differ by 3% in length but only by 1% in bits.
- The leftover spread is not random:
  - Abjads, scripts that leave out vowels, come out low: Hebrew -20%, Arabic -10%, Syriac -20%. The writing
    system drops information that readers fill in themselves.
  - Nepali and Marathi come out high (+20%). Hindi, in the same script, sits at the median, so this is the
    translation and not the script. Marathi and Nepali also pair in all four embedding instruments, which points
    to a shared translation lineage.
  - Without abjads, abugidas and that pair, the spread is 5.6% for bits against 25.5% for characters.
- NT set (84 languages, many of them small indigenous-language NTs): bits spread 10%, characters 29%. The
  highest are tone-marked orthographies (Chinantec, Cakchiquel +30%).

## 2. The "cooling curve"

When the character model may look further back, the spread between languages shrinks:

| order | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| bits per verse (median) | 552 | 420 | 336 | 273 | 242 | 231 | 230 |
| spread across languages | 17.6% | 16.2% | 16.0% | 12.0% | 8.8% | 7.2% | 7.2% |

- Languages look very different letter by letter. Given about five characters of context, they converge on a
  similar rate.
- NT curves rise again at order 6, because there is too little text to support that order. So order 6 is the
  ceiling for this instrument.

## 3. Word order and word forms trade off (a known law, reproduced)

- Shuffling the words inside each verse adds a median 20% of bits (character model), or 28% with lzma. That
  increase is the information carried by word order.
- Order information is strongly anti-correlated with the entropy of the word distribution: Spearman -0.81
  (CI -0.89 to -0.66), and -0.84 in the NT set.
  - Languages with rich word forms carry little in order: Finnish 21, Latin 22, Lithuanian 24, Hebrew 26 bits
    per verse.
  - Languages that build with separate words carry much more: Thai 190, Japanese 166, Malayalam 154.
- Because of this trade-off, the total is conserved better than either part. The spread is 6.6% with order and
  14.5% after word order is removed.
- This reproduces Koplenig et al. (2017) on about 1,000 languages. It's our first calibration against a published
  result, and it passes.

## 4. Chapters: where the information sits is shared by every language

- Bits per character by chapter (density) follows the same profile in every language. Two random halves of the
  languages agree at Spearman 0.98 (NT 0.99). 68% of the chapter-by-language variance in density is shared
  (NT 74%).
- Densest chapters: the genealogies and name lists (1CH.1-8, GEN.36, JOS.19) and Song of Songs.
- Sparsest chapters: Chronicles passages that repeat Kings (2CH.5, 6, 10, 18), ISA.39 (it repeats 2KI.20) and
  2KI.25 (repeated in JER.52).
- In the NT the sparsest are Mark and the passages Matthew shares with it, and the densest are Revelation, Luke 3
  (a genealogy) and Acts 27 (the shipwreck).
- So a plain compressor finds the synoptic parallels and the Chronicles/Kings duplication in every language.

## 5. Link to Atlas: geometry disagreement follows content, not packaging

Length-controlled consensus disagreement comes from the four embedding instruments.

| per-chapter measure | partial Spearman (length) | + source names |
|---|---|---|
| spread of bits across languages | -0.03 (CI -0.10 to 0.02) | -0.02 |
| shared density (how surprising the content is) | +0.22 (CI 0.17 to 0.28) | +0.18 |
| spread of density across languages | -0.17 | -0.18 |

NT: +0.20 for shared density, and -0.14 (CI includes 0) for the spread of bits.

- The chapters that embeddings find hard to translate are not the ones where languages package information
  differently.
- They are the ones whose content is surprising in every language. This fits the earlier finding that rare
  source lemmas and OT quotations predict disagreement. It also fits the layer picture, where language contributes
  about nothing and content contributes a lot.

## What this adds

- A first conserved quantity: about 230 bits per verse, within 6-7%.
- A first stat-mech style curve, the cooling curve.
- A first calibration against a known law.
- A probe-free, instrument-free check on the Atlas chapter results.
- Writing system turns out to be a real layer for this measure. Abjads drop 10-20%.

## Caveats

- Bits are bits of written text, not of the language. Orthography (vowels, tone marks, spacing) moves the totals.
- Word counts and word entropy depend on tokenisation (Chinese, Japanese, Thai and Vietnamese use the
  corpus -tok files).
- A character model at order 6 is a ceiling. A stronger model (PPM or a small neural model) would lower every
  number. The test is the spread, not the level.
- The Hebrew and Greek files here are not the original texts with vowels, so the "originals carry fewer bits"
  idea can't be tested with them.
