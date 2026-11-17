#!/usr/bin/env bash
# End to end in one command: environment -> offline tests -> pipeline -> verification -> lab notebook.
#   bash run_all.sh            # normal run (reuses the HTTP cache if it exists)
#   bash run_all.sh --fresh    # clear the cache first: what a reviewer does on a fresh clone
# Any extra arguments are passed to data_pipeline_ly.py.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "== 1/4 offline tests (failure modes, parsing, reproducibility)"
python -m pytest -q

echo "== 2/4 data pipeline (first run: about 5-10 min of API calls; later runs: seconds)"
python data_pipeline_ly.py "$@"

echo "== 3/4 check outputs against provenance_ly.json"
python data_pipeline_ly.py --verify

python - <<'PY'
import json, sys
p = json.load(open("provenance_ly.json", encoding="utf-8"))
failed = {d["name"]: d["failed_after_retries"] for d in p["datasets"] if d.get("failed_after_retries")}
if failed:
    print("WARNING: requests that failed after all retries:", failed)
    print("         run `bash run_all.sh` again later to fill in those fields (cached data is reused).")
if p["coverage"]["S669"]["with_alphafold_plddt"] == 0:
    sys.exit("STOP: no AlphaFold features were downloaded (AlphaFold DB unreachable?). "
             "Check the internet connection and run `bash run_all.sh` again.")
PY

echo "== 4/4 lab notebook (executed in place, outputs saved)"
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 lab_notebook_ly.ipynb

echo
echo "Done. Deliverables: data_pipeline_ly.py, provenance_ly.json, lab_notebook_ly.ipynb, README.md"
echo "Figure: results/first_result_vs_baseline_ly.png   Log: logs/pipeline_ly.log"
