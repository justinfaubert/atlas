#!/usr/bin/env bash
# Cross-text concept geometry on your GPU. Run from anywhere:  bash experiments/texts/run_texts.sh
# Fetches the library (10 Bible translations, NLTK Gutenberg set, 16 more Gutenberg books, 4 new texts),
# then measures concept geometry with 2 count-based instruments, 4 trained models and 1 untrained model.
# Expect roughly 30-60 minutes on an RTX 3080. Result: runs/texts_en_results.zip (upload that).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -c "import transformers" 2>/dev/null || pip install transformers
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
mkdir -p runs
python experiments/texts/fetch.py --out data/texts/en \
  --gutenberg 1228 1497 84 76 1661 1400 1727 2680 205 345 3207 3300 2591 4363 2800 30155
python experiments/texts/concepts.py --texts data/texts/en --out runs/texts_en \
  --probes-file experiments/texts/probes_en.txt --occ-cap 200 \
  --instruments svd ppmi hf:bert-base-uncased hf-random:bert-base-uncased hf:roberta-base hf:gpt2 \
                hf:sentence-transformers/all-mpnet-base-v2 2>&1 | tee runs/texts_en.log
cp data/texts/en/catalogue.csv runs/texts_en/
(cd runs && zip -qr texts_en_results.zip texts_en texts_en.log)
echo "Done: runs/texts_en_results.zip"
