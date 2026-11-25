#!/usr/bin/env bash
# English, French and German: Bibles and European Parliament debates, read by multilingual models.
# Run from anywhere:  bash experiments/texts/run_multi.sh
# Roughly 30-60 minutes on an RTX 3080. Result: runs/multi_results.zip (upload that).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -c "import transformers" 2>/dev/null || pip install transformers
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
mkdir -p runs
python experiments/texts/fetch_multi.py --out data/texts
INST="hf:xlm-roberta-base hf-random:xlm-roberta-base hf:bert-base-multilingual-cased hf:sentence-transformers/LaBSE"
for L in en fr de; do
  python experiments/texts/concepts.py --texts data/texts/$L --out runs/multi_$L \
    --probes-file experiments/texts/probes/$L.txt --occ-cap 200 --sizes 20000 50000 100000 \
    --instruments svd ppmi $INST 2>&1 | tee runs/multi_$L.log
done
bash experiments/texts/compare_multi.sh
