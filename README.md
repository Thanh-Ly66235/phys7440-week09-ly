# Assignment 9: Project Data Pipeline (AF-ΔΔG)

**Thanh Ly**, PHYS 7440 GenAI in Physics Research

**Project.** I ask whether the AlphaFold structure of the wild-type protein tells me anything reliable about a missense variant. The AI component, **AF-ΔΔG**, predicts the folding free-energy change ΔΔG (kcal/mol) of a point mutation from AlphaFold DB features at the mutated residue. It is compared with my Week 8 sequence-only baseline, PhysChem-Ridge (RMSE = 1.54 kcal/mol on S669), under the evaluation protocol of Assignment 8. This folder adds the **data ingestion layer**: one module that fetches, normalises, caches and documents every number the model is trained and scored on.

## Deliverables

| File | Assignment step |
|---|---|
| `data_pipeline_ly.py` | 1–2: ingestion module. It fetches from 2 dataset files and 4 APIs, normalises units and identifiers, caches results in SQLite and Parquet, and handles both failure modes. Every function has a docstring. |
| `provenance_ly.json` | 3: provenance record, **written by the pipeline**. For every dataset it gives the source URL or API endpoint, the access timestamps, the version or commit hash, the transformations applied, and response fingerprints. It also records the outputs' content hash, the schema and the sentinels. |
| `lab_notebook_ly.ipynb` | 4: lab notebook. It contains raw API output, the normalised version, physical sanity checks, the first AF-ΔΔG result and the comparison plot against the Week 8 baseline on all 669 S669 variants, plus a 10-variant close-up. |
| `README.md` | This file: how to run the pipeline end to end. |

Supporting files: `tests/test_data_pipeline_ly.py` contains 30 offline tests, `run_all.sh` runs everything in one command, and `requirements.txt` lists the dependencies. Several files come from Assignment 8 unchanged: `features_ly.py`, `scoring_ly.py`, `eval_harness.py` and `ground_truth_ly.json`. The frozen Week 8 baseline outputs are in `baseline/`.

## Requirements and installation

**System:** Python ≥ 3.10 and internet access to GitHub, EBI (`alphafold.ebi.ac.uk`, `www.ebi.ac.uk`) and UniProt (`rest.uniprot.org`). It was tested on Fedora 44 with Python 3.14.10. On Fedora, `sudo dnf install python3 python3-pip` provides Python if it is missing.

**Python packages** (listed in `requirements.txt`, installed automatically by `run_all.sh`):

| Package | Required version | Tested with | Used for |
|---|---|---|---|
| `httpx` | ≥ 0.27, < 1 | 0.28.1 | HTTP client of the pipeline (same library as the Assignment 7 MCP server) |
| `tenacity` | ≥ 8.2 | 9.2.1 | retries with exponential backoff (failure mode a) |
| `numpy` | ≥ 1.24 | 2.5.5 | numerics, contact numbers |
| `pandas` | ≥ 2.0 | 3.1.1 | tables and normalisation |
| `pyarrow` | ≥ 14 | 26.0.1 | Parquet output of the normalised table |
| `scikit-learn` | ≥ 1.3 | 1.9.1 | Week 8 baseline (ridge) and AF-ΔΔG (gradient boosting), notebook |
| `matplotlib` | ≥ 3.7 | 3.11.2 | notebook figures |
| `pint` | ≥ 0.23 | 0.26.2 | units in `eval_harness.py` (Meeting 1 harness), notebook |
| `pytest` | ≥ 8 | 9.2.0 | offline tests in `tests/` |
| `nbconvert` | ≥ 7 | 7.17.2 | runs the notebook from the command line |
| `ipykernel` | ≥ 6 | 7.4.0 | Jupyter kernel for the notebook |

Everything else used (`sqlite3`, `zlib`, `hashlib`, `json`, `logging`, `argparse`) is in the Python standard library. To open the notebook interactively, also install JupyterLab: `python -m pip install jupyterlab`, then `jupyter lab`.

**Installation by hand** (this is what `run_all.sh` does):

```bash
cd ~/Downloads/"Assignment 9"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The package versions of each pipeline run are recorded in `provenance_ly.json → pipeline.packages`. The first cell of the notebook prints the versions it ran with.

## Running the pipeline end to end

From this folder:

```bash
bash run_all.sh
```

The script does the following:

1. It creates `.venv` and installs `requirements.txt`.
2. It runs the offline tests.
3. It runs the pipeline. The first run makes about 850 polite API calls (about 10 min); later runs are served from the cache in seconds.
4. It checks the outputs against `provenance_ly.json`.
5. It executes the notebook in place, so the outputs are saved in it.

The same steps by hand:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q                                   # offline tests: failure modes, parsing, reproducibility
python data_pipeline_ly.py                            # fetch -> normalise -> cache -> provenance_ly.json
python data_pipeline_ly.py --verify                   # outputs match the provenance record (no network)
jupyter nbconvert --to notebook --execute --inplace lab_notebook_ly.ipynb   # or open it in Jupyter and Run All
```

Pipeline options:

| Option | Effect |
|---|---|
| `--fresh` | Clears the HTTP cache and downloads everything again, the way a reviewer would on a fresh clone. It then reports whether the content hash is identical to the one recorded in the existing `provenance_ly.json`. |
| `--offline` | Never uses the network. Cache misses become sentinels. |
| `--verify` | Recomputes the content hash of the normalised table and the dataset SHA-256s and compares them with `provenance_ly.json`. Exit code 1 on mismatch. |
| `--contact EMAIL` | Adds an e-mail address to the User-Agent (polite for EBI). |
| `--max-attempts N` | Attempts per request (≥ 4, default 5). |
| `--timeout S` | Per-request timeout (default 30 s). |
| `-v` | Also prints every missing-field event. |

**Before submitting**, check that the run produced:

- `provenance_ly.json`;
- `data/processed/variants_ly.parquet`;
- `results/first_result_vs_baseline_ly.png`;
- a notebook whose cells all have outputs.

Also check that the `network:` line near the end of the pipeline log reports `0 failed after retries`. If some requests failed, run `bash run_all.sh` again; only the missing pieces are downloaded. If the course expects a Git repository, run `git init && git add -A && git commit -m "Assignment 9"` in this folder. The `.gitignore` already keeps out the cache, the logs and `.venv`.

**Recommended final run before submitting:** `bash run_all.sh && bash run_all.sh --fresh`. The first run fills any gaps. The second clears the cache, downloads everything again as a fresh clone would, and records in `provenance_ly.json` whether the result is identical (`reproducibility.identical_to_previous`).

**Checking reproducibility from a fresh clone:** `python data_pipeline_ly.py --fresh`. The last log line says `REPRODUCIBLE: content hash identical to the previous provenance record`, or warns that the hash differs.

## What the pipeline does

```
S669 (GitHub, pinned commit) ──┐                       ┌─ AlphaFold DB /api/prediction/{acc}  (record, UniProt sequence)
                               ├─ normalise ─ numbering ┤  AlphaFold DB files v6: pLDDT JSON, PDB model, AlphaMissense CSV
S2648 (GitHub, pinned commit) ─┘   units      (SIFTS)   └─ UniProt REST FASTA (fallback sequence)
                                   IDs          │
                          PDBe SIFTS /mappings/uniprot/{pdb}
        -> features per variant -> data/processed/variants_ly.parquet (+ .csv) -> provenance_ly.json
```

| Source | Role | Version pin |
|---|---|---|
| S669 (Pancotti et al. 2022, doi:10.1093/bib/bbab555), UniProt-mapped file from `compbiomed-unito/JanusDDG` | test set, 669 variants | commit `7ef53fd` in the URL + SHA-256 `d6e82c53…` |
| S2648 (Dehouck et al. 2009, doi:10.1093/bioinformatics/btp445), `protddg-bench` | training set, 2648 variants | commit `72a5f30` in the URL + SHA-256 `e708b274…` |
| PDBe SIFTS API (`/pdbe/api/…`, fallback `/pdbe/graph-api/…`) | PDB chain and numbering → UniProt accession and numbering (S2648) | live API; SHA-256 fingerprint of all responses |
| AlphaFold DB API and files | model record, per-residue pLDDT, coordinates (→ burial), AlphaMissense | model files pinned to **v6** (as in Assignment 7) |
| UniProt REST | reference sequence when AlphaFold DB has no model | live, used only to validate numbering |
| GitHub commits API | latest upstream commit of each dataset file (provenance only) | — |

**Normalisation into one internal schema** (full list with units and sentinels in `provenance_ly.json → schema`):

- **Identifiers.** S669 IDs are cleaned (`P04517WTE2143G` → `P04517`). PDB-only chains get `MISSING`. S2648 PDB chains are mapped to UniProt accessions with SIFTS.
- **Residue numbering.** S2648 PDB author numbers are converted to UniProt numbers with the SIFTS segment offset. Every wild-type residue is checked against the UniProt sequence. When the segment offset fails, a chain-wide offset vote is used instead (≥ 3 sites, ≥ 80 % agreement). Insertion-code residues are left unmapped. The method used for each variant is stored in `numbering_map`, and the result of the check in `wt_check`.
- **Units and signs.** ΔΔG is stored in kcal/mol (a kJ/mol source would be converted by 1/4.184). The convention ΔΔG = ΔG<sub>unfold</sub>(mut) − ΔG<sub>unfold</sub>(wt), with < 0 destabilising, is checked for both files. Temperatures are converted from °C to K.
- **Structural features** in UniProt numbering:
  - pLDDT at the site and its ±5-residue mean;
  - burial, as the number of Cβ atoms within 10 Å in the AlphaFold model;
  - relative position in the chain;
  - AlphaMissense score and class of the exact substitution, and the mean score at the site.

The 10 sequence descriptors of the Week 8 baseline are computed from `source_mutation` with `features_ly.py`, exactly as in Week 8.

## Failure handling

**(a) API timeout, rate limit or server error.** Every request goes through `tenacity.Retrying`:

- up to **5 attempts (4 retries)**, with exponential backoff of 1, 2, 4 and 8 s after a timeout, connection error or HTTP 5xx;
- a **rate limit (HTTP 429)** waits longer, 5, 10, 20 and 40 s, because rate-limit windows last tens of seconds;
- a `Retry-After` header is honoured, and every wait is capped at 60 s;
- politeness: at most one request every 0.25 s per host. After an HTTP 429 that host's interval is doubled, up to 2 s, for the rest of the run (adaptive throttling).

Each retry is logged. If all attempts fail, the request raises `UpstreamError`, which the caller catches. After 5 consecutive failed requests to one host, the remaining requests to that host fail fast for the rest of the run (circuit breaker), so a full outage costs seconds rather than hours. The affected fields then become sentinels and are logged, and the run continues. `failed_after_retries` in the provenance counts these requests, and re-running fills them in, because failures are never cached.

**(b) Missing or null fields.** Every value that cannot be obtained is replaced by a **documented sentinel**:

- `-1` for numeric fields whose real values are never negative (pLDDT, AlphaMissense, contact count, residue number, pH, temperature in K);
- `"MISSING"` for strings;
- `NaN` for ΔΔG, because −1 kcal/mol is a real ΔΔG value.

Each substitution is logged with the variant, the field, the reason and the **source URL** in three places: `logs/pipeline_ly.log`, `data/processed/missing_fields_ly.csv`, and the row's own `missing_fields` column. Typical causes are a protein with no AlphaFold model, a non-human protein (AlphaMissense covers human proteins only), a PDB chain without a SIFTS mapping, a null pLDDT, and pH or temperature absent from S669. The notebook turns sentinels into NaN with `dp.sentinels_to_nan()` before any calculation.

Both failure modes are tested offline with `httpx.MockTransport` (`tests/`):

- a timeout followed by success;
- HTTP 429;
- HTTP 503 until the retries are exhausted;
- a complete AlphaFold DB outage, after which the pipeline finishes with sentinels;
- a null pLDDT;
- a non-human protein;
- a missing AlphaFold model.

## Caching and reproducibility

- **Raw responses.** `data/cache/http_cache_ly.sqlite` (git-ignored) is keyed by URL. It stores each body zlib-compressed, with the HTTP status, the SHA-256 and the original access time. Only definitive answers (200, 400, 404) are cached.
- **Normalised table.** `data/processed/variants_ly.parquet`, with a CSV copy for diffs and peer review.
- **Content hash.** `content_sha256` in the provenance is the SHA-256 of a canonical CSV serialisation, so it does not depend on the Parquet library version. Rebuilding the table from the same cache on a second machine (different Python version) gave a byte-identical content hash. Dataset files are pinned to commits and SHA-256-checked, and a changed file stops the run. AlphaFold files are pinned to v6. Rows are written in a fixed order, and floats are rounded at the source precision.
- **What could legitimately change.** SIFTS mappings are served live, and AlphaFold DB could retire v6. Either change would alter the per-source `responses_sha256` and the content hash. `--fresh` reports it, and `provenance_ly.json → reproducibility.differences_from_previous_table` lists which columns changed in how many rows, so an upstream data change can be told apart from a bug.
- **Baseline check.** The notebook refits the Week 8 baseline from the pipeline's table and checks that its predictions equal the frozen Week 8 predictions to machine precision.

## Output layout

```
provenance_ly.json                  provenance record (written by the pipeline)
data/raw/                           verbatim copies of S669 and S2648 (SHA-256 checked)
data/processed/variants_ly.parquet  normalised table, 3317 rows (669 test + 2648 train)
data/processed/variants_ly.csv      same, as CSV
data/processed/missing_fields_ly.csv  every substituted value with its source URL and reason
data/cache/http_cache_ly.sqlite     raw HTTP responses (git-ignored)
logs/pipeline_ly.log                run log: progress, retries, one line per missing field
results/                            notebook outputs: comparison figure, predictions, metrics JSON
baseline/                           frozen Week 8 baseline predictions and metrics
```

## Lab notebook

`lab_notebook_ly.ipynb` reads only the pipeline outputs and the cache, so it runs offline. Its sections are:

1. **Provenance overview.**
2. **Raw API output.** The verbatim AlphaFold DB record, pLDDT JSON, PDB atoms and AlphaMissense rows for RAF1 L78A, my Meeting 1 ground-truth item, and a raw SIFTS response.
3. **The normalised version** of the same records.
4. **Physical sanity checks** on S2648 only: ΔΔG vs burial, and pLDDT at the sites.
5. **First result.**
   - The Week 8 baseline rebuilt from the pipeline data, identical to Week 8.
   - AF-ΔΔG (gradient boosting on the 10 baseline descriptors plus 6 AlphaFold features), with a sequence-only control.
   - Scoring once on S669: RMSE with protein-bootstrap CIs, paired ΔRMSE, and the Meeting 1 harness.
   - The **comparison plot**: a parity plot of all 669 variants, 10 individual variants including both ΔΔG ground-truth items, RMSE with CIs, and RMSE by burial class.
6. **Results and discussion**:
   - a summary generated from the computed numbers;
   - a written discussion of the first result;
   - a cell that checks every number quoted in the discussion against the outputs;
   - limitations and next steps.

## Relation to Assignments 7 and 8

The pipeline calls the same AlphaFold DB endpoints as my Assignment 7 MCP server (`../Assignment 7/mcp_server_ly.py`). It picks the AlphaFold record with the same rule and pins the same model version. The tests use the AlphaFold DB responses recorded for that server (`tests/fixtures/afdb`). They check that the pipeline extracts exactly the pLDDT and AlphaMissense values the server reported:

- HBB E7V: pLDDT 96.56, AlphaMissense 0.2229;
- TP53 R175H: pLDDT 96.62, AlphaMissense 0.9857.

The test set, metric, success threshold (RT ln 10 = 1.36 kcal/mol), cross-validation folds, scoring code and harness all come from Assignment 8 without changes.
