#!/usr/bin/env bash
# The message as what two translations share: NLLB-200 scores every held-out verse given its real source verse,
# given a different verse, and given nothing. Run from anywhere:  bash experiments/information/run_shared.sh
# Downloads the 600M NLLB model (about 2.5 GB) on first use. Roughly 30-60 minutes on an RTX 3080.
# Resumable: rerun to continue. Results stay in runs/info_nllb (nothing to upload).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -c "import transformers, sentencepiece" 2>/dev/null || pip install transformers sentencepiece
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
mkdir -p runs/info_nllb
python experiments/information/gpu_shared_nllb.py --bibles data/bible --out runs/info_nllb "$@" 2>&1 | tee -a runs/info_nllb/run.log
echo "Done. Results are in runs/info_nllb."
