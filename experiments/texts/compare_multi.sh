#!/usr/bin/env bash
# Re-run only the cross-language comparison on finished runs/multi_{en,fr,de} (seconds, no GPU), then zip.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
INST="hf:xlm-roberta-base hf-random:xlm-roberta-base hf:bert-base-multilingual-cased hf:sentence-transformers/LaBSE"
: > runs/multi_compare.log
for I in svd ppmi $INST; do
  python experiments/texts/compare_languages.py --run en=runs/multi_en --run fr=runs/multi_fr --run de=runs/multi_de \
    --instrument "$I" --out runs/multi_compare | tee -a runs/multi_compare.log
done
for I in $INST; do
  python experiments/texts/compare_languages.py --run en=runs/multi_en --run fr=runs/multi_fr --run de=runs/multi_de \
    --instrument "$I" --text-effect --out runs/multi_compare | tee -a runs/multi_compare.log
done
(cd runs && rm -f multi_results.zip && zip -qr multi_results.zip multi_en multi_fr multi_de multi_compare multi_*.log)
echo "Done: runs/multi_results.zip"
