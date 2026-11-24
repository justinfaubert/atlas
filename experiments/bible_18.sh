#!/usr/bin/env bash
# First fibre bundle experiment: the Bible in 18 languages, chapter level, LSA per language.
# Data: https://github.com/christos-c/bible-corpus (bibles/<Language>.xml) in $1.
set -euo pipefail
BIBLE=${1:-data/bible}
OUT=${2:-reports/bible_18_languages.html}
atlas bundle --bible "$BIBLE" \
  --langs English French German Spanish Latin Greek Russian Finnish Hungarian Turkish Arabic Hebrew Hindi Indonesian Vietnamese Chinese Japanese Korean \
  --level chapter --embed lsa:200 --k 32 --null-reps 5 \
  --loop Chinese Japanese French English --snippet-view English \
  --title "Fibre bundle over the Bible in 18 languages" \
  --finding "Going around the Chinese, Japanese, French, English loop twice is the spinor test: 180-degree flips cancel on the second trip, while the genuine twist grows. The flips sit in the weakest, least determined directions, so they are most likely noise; the twist is not." \
  --finding "The most twisted chapters are mostly poetry and prophecy (Zechariah, Hosea, Psalms, Isaiah, Job, Lamentations); the least twisted are mostly law, genealogy and plain narrative (Leviticus, Numbers, Chronicles, Mark, Luke). Proverbs is an exception worth a look." \
  --caveat "Each language's space is built from its own character statistics (LSA), not a neural model, so part of what differs between views is spelling and word formation rather than meaning. Rerunning with neural models on your machine is the real test." \
  --caveat "There is one translation per language, so language and translator are mixed up. Closeness may reflect translation tradition (which source a Bible was translated from) as much as language family. Adding several translations per language separates the two." \
  --caveat "Directions with little variance are poorly determined, and flips there are noise. That is why the headline measure is twist, which weights each direction by how much of the text lives there, rather than the Wilson loop trace." \
  --out "$OUT"
