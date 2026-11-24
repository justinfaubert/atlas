#!/usr/bin/env bash
# Scale bundle: Bible chapters at four scales (word, verse, paragraph, whole) in 12 languages, LSA per language.
set -euo pipefail
BIBLE=${1:-data/bible}
OUT=${2:-reports/scales_12_languages.html}
atlas scales --bible "$BIBLE" \
  --langs English German French Spanish Greek Russian Finnish Hungarian Arabic Hebrew Chinese Japanese \
  --parent chapter --embed lsa:200 --block 5 --k 24 --null-reps 5 --snippet-view English \
  --title "Scale bundle: the Bible taken apart and put back together, in 12 languages" \
  --finding "In every language about nine tenths of a chapter's whole can be rebuilt from its words, verses and paragraphs. The tenth that can't is what we call binding." \
  --finding "The same chapters bind strongly in all 12 languages, even after removing chapter length and repetition, and shuffling chapters destroys the agreement. What a whole adds beyond its parts looks like a property of the text, not of a language." \
  --finding "Binding is concentrated. For most chapters the whole is close to the sum of its parts, while a handful bind far more than the rest. Those are mostly refrains and lists: Psalm 136's repeated refrain, the musicians of 1 Chronicles 25, the returnees of Ezra 2, the kings of Joshua 12, the offerings of Numbers 7. The agreement between languages is measured on ranks, so it doesn't rest on these few." \
  --finding "Scale and language interact most at the step from words to verses. Above that, between verses, paragraphs and wholes, the plaquette twist is about the size of the twist between languages." \
  --finding "The top three directions line up with the whole's own better than any later directions do, at every scale. The fall-off is gradual, though, with no sharp break after the third." \
  --caveat "Spaces are built from character statistics (LSA), so the word scale is crude, and for Chinese and Japanese a word is a single character. Neural models on your machine are the real test." \
  --caveat "Repetition is compressed when a whole is weighted as one document but not when its parts are averaged. That explains part of why refrains bind. The agreement between languages survives a type-token repetition control, but neural models treat repetition differently and should be checked." \
  --caveat "Paragraphs are blocks of five verses, not the translators' own paragraphs." \
  --out "$OUT"
