#!/usr/bin/env bash
# Dialogue vs narration: 18 novels split into quoted speech and narration, each part read at 20k words, through
# English models. Tests whether mode of address shapes concept geometry with author and story held fixed.
# Run from anywhere:  bash experiments/texts/run_dialogue.sh
# Roughly 20-40 minutes on an RTX 3080. Nothing to upload: results stay in runs/dialogue20.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -c "import transformers" 2>/dev/null || pip install transformers
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
mkdir -p runs
[ -f data/texts_kinds/brown-fiction.txt ] || python experiments/texts/fetch_kinds.py --out data/texts_kinds
python experiments/texts/dialogue_split.py --texts data/texts_fiction --fetch --kinds data/texts_kinds \
  --out data/texts_dialogue
python experiments/texts/concepts.py --texts data/texts_dialogue --out runs/dialogue20 --cap 20000 \
  --probes-file experiments/texts/probes_wide/en.txt --occ-cap 200 --sizes 10000 \
  --instruments svd ppmi hf:bert-base-uncased hf-random:bert-base-uncased hf:roberta-base \
                hf:sentence-transformers/all-mpnet-base-v2 2>&1 | tee runs/dialogue20.log
cp data/texts_dialogue/catalogue.csv runs/dialogue20/
python experiments/texts/dialogue_map.py --run runs/dialogue20 --out runs/dialogue20/map \
  --instruments svd ppmi hf:bert-base-uncased hf-random:bert-base-uncased hf:roberta-base \
                hf:sentence-transformers/all-mpnet-base-v2 > runs/dialogue20/map.log
echo "Done. Results are in runs/dialogue20."
