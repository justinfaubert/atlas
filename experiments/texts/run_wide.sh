#!/usr/bin/env bash
# Step 1b: the wider aligned concept set (123 concepts with verbs) in English, French and German, read by the
# multilingual models. Run from anywhere:  bash experiments/texts/run_wide.sh
# Roughly 1-2 hours on an RTX 3080. Nothing to upload: the results stay in runs/wide_{en,fr,de}.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -c "import transformers" 2>/dev/null || pip install transformers
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
mkdir -p runs
python experiments/texts/fetch_multi.py --out data/texts
INST="hf:xlm-roberta-base hf-random:xlm-roberta-base hf:bert-base-multilingual-cased hf:sentence-transformers/LaBSE"
for L in en fr de; do
  # an earlier fetch may have left the OCR-damaged GerBoLut file in data/texts/de: read the texts through a
  # folder of links that leaves it out
  W=data/texts_wide/$L; rm -rf "$W"; mkdir -p "$W"
  for f in data/texts/$L/*.txt; do case "$f" in *BoLut*) ;; *) ln -s "$(realpath "$f")" "$W/";; esac; done
  grep -v BoLut data/texts/$L/catalogue.csv > "$W/catalogue.csv"
  python experiments/texts/concepts.py --texts "$W" --out runs/wide_$L \
    --probes-file experiments/texts/probes_wide/$L.txt --occ-cap 200 --sizes 20000 100000 \
    --instruments svd ppmi $INST 2>&1 | tee runs/wide_$L.log
done
echo "Done. Results are in runs/wide_en, runs/wide_fr, runs/wide_de."
