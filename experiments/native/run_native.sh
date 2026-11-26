#!/usr/bin/env bash
# Native vs translated novels read by the multilingual models (the neural half of the native-texts test).
# Run from anywhere:  bash experiments/native/run_native.sh
# Downloads 62 public-domain novels (about 60 MB) from the GITenberg mirror, then reads them with three trained
# models and one untrained control. Roughly 2-3 hours on an RTX 3080. Results stay in runs/native_{en,fr,de}.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -c "import transformers" 2>/dev/null || pip install transformers
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
mkdir -p runs data
[ -f data/native/catalogue.csv ] || python experiments/native/fetch_native.py --out data/native
python - <<'PY'
import csv, os
from pathlib import Path
src = Path("data/native"); rows = list(csv.DictReader(open(src / "catalogue.csv")))
for L in ["en", "fr", "de"]:
    d = Path("data/native_by_lang") / L; d.mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        if r["language"] != L: continue
        t = d / f"{r['name']}.txt"
        if not t.exists(): os.symlink((src / f"{r['name']}.txt").resolve(), t)
        out.append({"name": r["name"], "kind": "native" if r["native"] == "1" else "translated"})
    with open(d / "catalogue.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "kind"]); w.writeheader(); w.writerows(out)
PY
INST="hf:xlm-roberta-base hf-random:xlm-roberta-base hf:bert-base-multilingual-cased hf:sentence-transformers/LaBSE"
for L in en fr de; do
  python experiments/texts/concepts.py --texts data/native_by_lang/$L --out runs/native_$L \
    --probes-file experiments/texts/probes_wide/$L.txt --occ-cap 200 --sizes 20000 100000 \
    --instruments svd ppmi $INST 2>&1 | tee runs/native_$L.log
done
echo "Done. Results are in runs/native_en, runs/native_fr, runs/native_de. Tell Claude 'done'."
