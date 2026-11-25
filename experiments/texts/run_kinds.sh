#!/usr/bin/env bash
# Step 2: map of kinds of text (145 English texts in 21 kinds, each read at 40k words) through English models.
# Run from anywhere:  bash experiments/texts/run_kinds.sh
# Roughly 1-2 hours on an RTX 3080. Nothing to upload: results stay in runs/kinds40.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -c "import transformers" 2>/dev/null || pip install transformers
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
mkdir -p runs
python experiments/texts/fetch_kinds.py --out data/texts_kinds
# Blake (7k words) and Burgess (16k) are too short for a 40k-word reading
mkdir -p data/texts_kinds_excluded
for f in poetry-blake children-burgess; do
  [ -f data/texts_kinds/$f.txt ] && mv data/texts_kinds/$f.txt data/texts_kinds_excluded/
done
grep -v "poetry-blake\|children-burgess" data/texts_kinds/catalogue.csv > data/texts_kinds/catalogue.tmp
mv data/texts_kinds/catalogue.tmp data/texts_kinds/catalogue.csv
python experiments/texts/concepts.py --texts data/texts_kinds --out runs/kinds40 --cap 40000 \
  --probes-file experiments/texts/probes_wide/en.txt --occ-cap 200 --sizes 20000 \
  --instruments svd ppmi hf:bert-base-uncased hf-random:bert-base-uncased hf:roberta-base \
                hf:sentence-transformers/all-mpnet-base-v2 2>&1 | tee runs/kinds40.log
cp data/texts_kinds/catalogue.csv runs/kinds40/
echo "Done. Results are in runs/kinds40."
