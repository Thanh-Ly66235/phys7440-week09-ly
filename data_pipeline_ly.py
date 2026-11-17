"""AF-ΔΔG data ingestion pipeline (PHYS 7440, Assignment 9) — Thanh Ly.

This module is the data layer of my final project. The project asks whether the
AlphaFold structure of the wild-type protein can predict the folding free-energy
change ΔΔG of a missense variant (see ``README.md`` and the Assignment 8
evaluation protocol). The module turns two experimental ΔΔG datasets plus the
structure and annotation services below into one normalised, cached table that
every later model is trained and scored on.

Sources (all public, no API keys)
---------------------------------
1. **S669** test set (file on GitHub; Pancotti et al. 2022, doi:10.1093/bib/bbab555),
   already in UniProt numbering. Pinned to a commit and SHA-256 checked.
2. **S2648** training set (file on GitHub; Dehouck et al. 2009,
   doi:10.1093/bioinformatics/btp445), in PDB author numbering. Pinned and checked.
3. **PDBe SIFTS API** ``/pdbe/api/mappings/uniprot/{pdb}``: PDB chain -> UniProt
   accession and residue-number segments (identifier normalisation of S2648).
4. **AlphaFold DB API** ``/api/prediction/{accession}`` plus the v6 model files
   (per-residue pLDDT JSON, PDB coordinates, AlphaMissense substitution table).
   These are the same endpoints my Assignment 7 MCP server uses.
5. **UniProt REST** ``/uniprotkb/{accession}.fasta``: fallback reference sequence,
   used only to validate residue numbering when AlphaFold DB has no model.
6. **GitHub commits API** (provenance only): the latest upstream commit of each
   dataset file, so that a silent upstream change shows up in the provenance.

Internal schema
---------------
One row per variant, written to ``data/processed/variants_ly.parquet`` (plus a CSV
copy). Every column, its unit and its sentinel are listed in :data:`SCHEMA`.
Conventions after normalisation:

* residues are numbered in **UniProt** coordinates (S2648 is mapped from PDB numbering);
* ``ddg_kcal_mol`` is ΔΔG = ΔG_unfold(mutant) − ΔG_unfold(wild type) in **kcal/mol**,
  so ΔΔG < 0 means destabilising (both sources already use this sign; it is checked);
* temperatures are in **kelvin**, pLDDT in [0, 100], AlphaMissense in [0, 1].

Failure handling (assignment step 2)
------------------------------------
(a) **Timeouts, rate limits and server errors**: every HTTP request goes through a
    ``tenacity.Retrying`` loop with up to 5 attempts (4 retries) and exponential
    backoff (1, 2, 4, 8 s, capped at 60 s) that honours a ``Retry-After`` header. If
    the last attempt also fails, :class:`UpstreamError` is raised and caught by the
    caller, which records the affected fields as missing instead of stopping the run.
(b) **Missing or null fields**: whenever a value cannot be obtained (no accession,
    no AlphaFold model, null pLDDT, non-human protein without AlphaMissense, residue
    outside the model, numbering mismatch, absent source column, failed request ...)
    :class:`MissingFieldLog` logs the field, the variant and the **source URL**, and the
    value is replaced by the documented sentinel (:data:`SENTINELS`): ``-1`` for
    numeric fields whose real values are never negative, ``"MISSING"`` for strings,
    and ``NaN`` for ΔΔG itself (−1 kcal/mol is a valid ΔΔG). Each row also lists its
    substituted fields in ``missing_fields``.

Caching and reproducibility
---------------------------
Raw HTTP bodies are cached in SQLite (``data/cache/http_cache_ly.sqlite``, zlib
compressed, keyed by URL, with the original access time and SHA-256), so a second
run makes no network calls. Dataset files are pinned to Git commits and checked
against SHA-256 digests; AlphaFold files are pinned to model version 6. The
normalised table has a content hash (SHA-256 of a canonical CSV serialisation)
that is written to ``provenance_ly.json``. A fresh clone must reproduce it:
``python data_pipeline_ly.py --fresh`` re-downloads everything and compares.

Usage
-----
    python data_pipeline_ly.py            # fetch (or reuse cache) -> normalise -> outputs + provenance
    python data_pipeline_ly.py --fresh    # ignore the cache (simulates a fresh clone) and compare hashes
    python data_pipeline_ly.py --offline  # never touch the network; cache misses become sentinels
    python data_pipeline_ly.py --verify   # check the outputs against provenance_ly.json, no network
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import platform
import re
import sqlite3
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
import pandas as pd
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt

PIPELINE_NAME = "af-ddg-data-pipeline"
PIPELINE_VERSION = "1.0.0"
AUTHOR = "Thanh Ly"
HERE = Path(__file__).resolve().parent

log = logging.getLogger("af_ddg")
missing_log = logging.getLogger("af_ddg.missing")   # one line per substituted field (log file only)

# ======================================================================================
# Configuration
# ======================================================================================

AFDB_BASE = "https://alphafold.ebi.ac.uk"
AFDB_MODEL_VERSION = 6                      # pinned model version (same as Assignment 7)
AFDB_PREDICTION = AFDB_BASE + "/api/prediction/{accession}"
AFDB_FILE = AFDB_BASE + "/files/AF-{accession}-F1-{kind}"
PDBE_SIFTS = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
PDBE_SIFTS_FALLBACK = "https://www.ebi.ac.uk/pdbe/graph-api/mappings/uniprot/{pdb_id}"   # same JSON format
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
GITHUB_COMMITS = "https://api.github.com/repos/{repo}/commits?path={path}&per_page=1"

BURIAL_RADIUS_A = 10.0       # contact-number radius (Å) between Cβ atoms (Cα for Gly)
PLDDT_WINDOW = 5             # ± residues for the local mean pLDDT
OFFSET_VOTE_MIN_SITES = 3    # distinct sites needed before a free offset search is trusted
OFFSET_VOTE_MIN_FRACTION = 0.8

UNIPROT_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")
MUTATION_RE = re.compile(r"^([A-Z])(-?\d+)([A-Z]?)([A-Z])$")     # wt, number, insertion code, mutant
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
AM_CLASS = {"LBen": "likely_benign", "Amb": "ambiguous", "LPath": "likely_pathogenic"}
DDG_UNIT_FACTORS = {"kcal/mol": 1.0, "kJ/mol": 1.0 / 4.184}       # -> kcal/mol


@dataclass(frozen=True)
class FileSource:
    """A frozen dataset file in a public Git repository, pinned to one commit."""

    name: str
    role: str
    repo: str
    commit: str              # commit pinned in the download URL
    file_commit: str         # last commit that changed the file (git log -1 -- path)
    file_commit_date: str
    path: str
    sha256: str              # expected digest (frozen in Assignment 8)
    local_name: str
    citation: str
    doi: str
    numbering: str
    ddg_unit: str            # unit of the ΔΔG column in the file
    ddg_sign: int            # +1 if the file already uses ΔΔG < 0 = destabilising
    temperature_unit: str | None

    @property
    def url(self) -> str:
        """Raw-file URL pinned to :attr:`commit` (immutable)."""
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{self.path}"


S669 = FileSource(
    name="S669", role="test",
    repo="compbiomed-unito/JanusDDG",
    commit="7ef53fde72c9b9b715b0a9f2f80910ebf927bd59",            # tag v1.0.1
    file_commit="01c1a5afe8f2cd28620e4df6ccf340e81f57f45e", file_commit_date="2025-09-10T15:49:08+02:00",
    path="data/s669_to_process.csv",
    sha256="d6e82c53c6554988bdfa0f5cf5e457b421c3bcdc279e3bbf9e5304dddf0efe51",
    local_name="s669_uniprot.csv",
    citation="Pancotti C. et al. (2022) Brief. Bioinform. 23, bbab555; UniProt-mapped file from JanusDDG (Barducci et al. 2025)",
    doi="10.1093/bib/bbab555", numbering="UniProt", ddg_unit="kcal/mol", ddg_sign=+1,
    temperature_unit=None,
)
S2648 = FileSource(
    name="S2648", role="train",
    repo="protddg-bench/protddg-bench.github.io",
    commit="72a5f300ca0f5076f7f6c644d58c2f7040e021ad",
    file_commit="56c4e1d54b5176cf26a1738bff0e731650e394bb", file_commit_date="2020-07-14T00:36:55+02:00",
    path="data/s2648.csv",
    sha256="e708b274574b02d8cc0b39efc0f09b6180ae211c714306ec05c685c86b55766c",
    local_name="s2648.csv",
    citation="Dehouck Y. et al. (2009) Bioinformatics 25, 2537 (protddg-bench release)",
    doi="10.1093/bioinformatics/btp445", numbering="PDB author numbering", ddg_unit="kcal/mol", ddg_sign=+1,
    temperature_unit="degC",
)
FILE_SOURCES = (S669, S2648)

# ======================================================================================
# Internal schema and sentinels
# ======================================================================================

SENTINEL_FLOAT = -1.0
SENTINEL_INT = -1
SENTINEL_STR = "MISSING"
SENTINELS = {
    "float": SENTINEL_FLOAT,
    "int": SENTINEL_INT,
    "str": SENTINEL_STR,
    "ddg_kcal_mol": "NaN",
    "why": ("-1 is used only for fields whose real values are never negative (pLDDT 0-100, "
            "AlphaMissense 0-1, contact counts >= 0, residue numbers >= 1, pH >= 0, T > 0 K), so it "
            "cannot be mistaken for data. ΔΔG uses NaN because -1 kcal/mol is a real ΔΔG. Every "
            "substituted field is also listed in the row's 'missing_fields' column and in "
            "data/processed/missing_fields_ly.csv with its source URL."),
}

# column -> (dtype, unit, description); the sentinel follows from the dtype (see SENTINELS)
SCHEMA: dict[str, tuple[str, str | None, str]] = {
    "variant_id": ("str", None, "stable ID '<dataset>:<source row, 4 digits>'"),
    "dataset": ("str", None, "S669 or S2648"),
    "split": ("str", None, "test (S669) or train (S2648)"),
    "source_row": ("int", None, "0-based row number in the source file"),
    "source_id": ("str", None, "identifier as written in the source (S669 ID column; S2648 PDB_CHAIN)"),
    "source_mutation": ("str", None, "mutation as written in the source, in source numbering"),
    "group": ("str", None, "clustering key: S669 protein, S2648 sequence cluster CLID"),
    "protein": ("str", None, "protein key: cleaned S669 ID, or S2648 PDB_CHAIN"),
    "uniprot_accession": ("str", None, "UniProt accession of the protein"),
    "wt": ("str", None, "wild-type residue (one-letter)"),
    "mt": ("str", None, "mutant residue (one-letter)"),
    "position_uniprot": ("int", None, "residue number in UniProt numbering"),
    "mutation_uniprot": ("str", None, "mutation in UniProt numbering, e.g. S11A"),
    "numbering_map": ("str", None, "how the UniProt position was obtained (status column, never a sentinel)"),
    "wt_check": ("str", None, "wild type vs reference sequence: match / mismatch / unverified (status column)"),
    "ddg_kcal_mol": ("float", "kcal/mol", "ΔΔG = ΔG_unfold(mut) − ΔG_unfold(wt); < 0 destabilising (sentinel NaN)"),
    "ph": ("float", None, "pH of the measurement"),
    "temperature_k": ("float", "K", "temperature of the measurement"),
    "af_entry_id": ("str", None, "AlphaFold DB entry, e.g. AF-P00149-F1"),
    "af_model_version": ("int", None, "AlphaFold DB model version (latestVersion of the entry)"),
    "af_model_created": ("str", None, "modelCreatedDate of the AlphaFold DB entry"),
    "plddt": ("float", "pLDDT (0-100)", "AlphaFold per-residue confidence at the site"),
    "plddt_window": ("float", "pLDDT (0-100)", f"mean pLDDT over ±{PLDDT_WINDOW} residues"),
    "burial_cn10": ("int", "count", f"Cβ atoms (Cα for Gly) within {BURIAL_RADIUS_A:g} Å of the site's Cβ in the AF model"),
    "rel_position": ("float", "fraction", "position / sequence length"),
    "am_pathogenicity": ("float", "probability (0-1)", "AlphaMissense pathogenicity of this exact substitution"),
    "am_class": ("str", None, "AlphaMissense class: likely_benign / ambiguous / likely_pathogenic"),
    "am_position_mean": ("float", "probability (0-1)", "mean AlphaMissense score of the 19 substitutions at the site"),
    "missing_fields": ("str", None, "';'-separated list of fields replaced by sentinels ('' if none)"),
}
COLUMNS = list(SCHEMA)
STATUS_COLUMNS = {"variant_id", "dataset", "split", "source_row", "numbering_map", "wt_check", "missing_fields"}
AF_FIELDS = ["af_entry_id", "af_model_version", "af_model_created", "plddt", "plddt_window",
             "burial_cn10", "rel_position", "am_pathogenicity", "am_class", "am_position_mean"]
AM_FIELDS = ["am_pathogenicity", "am_class", "am_position_mean"]
NUMERIC_SENTINEL_COLUMNS = [c for c, (t, _, _) in SCHEMA.items()
                            if t in ("float", "int") and c not in STATUS_COLUMNS and c != "ddg_kcal_mol"]


def sentinel_for(column: str) -> Any:
    """Return the documented sentinel value for one schema column."""
    if column == "ddg_kcal_mol":
        return float("nan")
    return {"float": SENTINEL_FLOAT, "int": SENTINEL_INT, "str": SENTINEL_STR}[SCHEMA[column][0]]


def sentinels_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of a normalised table with the numeric sentinels (-1) turned into NaN.

    Use this before modelling or plotting so that sentinels are never treated as
    data. String sentinels ('MISSING') are left as they are.
    """
    out = df.copy()
    for c in NUMERIC_SENTINEL_COLUMNS:
        if c in out:
            out[c] = out[c].astype(float).mask(out[c] == SENTINEL_INT)
    return out


def load_variants(root: Path | str = HERE) -> pd.DataFrame:
    """Read the normalised table written by :func:`run` (sentinels kept)."""
    return pd.read_parquet(Paths(Path(root)).parquet)


# ======================================================================================
# Failure mode (b): missing / null fields
# ======================================================================================


@dataclass
class MissingFieldLog:
    """Collects every field that was replaced by a sentinel, with its source URL.

    Each event is written to the ``af_ddg.missing`` logger (the log file) and kept in
    memory so that it can be saved as ``missing_fields_ly.csv`` and summarised in the
    provenance record. ``variant_id='*'`` marks a whole-dataset event (e.g. a column
    that the source file does not have).
    """

    events: list[dict[str, str]] = field(default_factory=list)

    def record(self, variant_id: str, fields: Iterable[str], source_url: str, reason: str) -> None:
        """Log that ``fields`` of ``variant_id`` were substituted, and why."""
        fields = list(fields)
        for f in fields:
            self.events.append({"variant_id": variant_id, "field": f, "source_url": source_url, "reason": reason})
        missing_log.warning("missing %s for %s: %s [source: %s]", ",".join(fields), variant_id, reason, source_url)

    def as_frame(self) -> pd.DataFrame:
        """All events as a DataFrame (columns variant_id, field, source_url, reason)."""
        return pd.DataFrame(self.events, columns=["variant_id", "field", "source_url", "reason"])

    def summary(self) -> dict[str, int]:
        """Number of logged substitutions per field (row-level events only)."""
        counts: dict[str, int] = defaultdict(int)
        for e in self.events:
            if e["variant_id"] != "*":
                counts[e["field"]] += 1
        return dict(sorted(counts.items()))


# ======================================================================================
# HTTP layer: SQLite cache and failure mode (a), tenacity retries with exponential backoff
# ======================================================================================


class UpstreamError(Exception):
    """A request that could not be completed (after all retries, or an offline cache miss)."""

    def __init__(self, url: str, reason: str):
        """Store the failing URL and a human-readable reason."""
        super().__init__(f"{reason} [{url}]")
        self.url, self.reason = url, reason


class RetryableHTTPStatus(Exception):
    """HTTP 429 or 5xx: worth retrying. Carries the server's Retry-After, if any."""

    def __init__(self, url: str, status: int, retry_after: float | None):
        """Store the URL, the status code and the parsed Retry-After seconds."""
        super().__init__(f"HTTP {status} from {url}")
        self.url, self.status, self.retry_after = url, status, retry_after


RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, RetryableHTTPStatus)


@dataclass
class Response:
    """One HTTP response, live or from the cache."""

    url: str
    status: int
    body: bytes
    content_type: str
    fetched_at: str          # ISO-8601 UTC time of the original network access
    sha256: str
    from_cache: bool
    attempts: int = 1

    def text(self) -> str:
        """Body decoded as UTF-8 (invalid bytes replaced)."""
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        """Body parsed as JSON; raises UpstreamError if it is not valid JSON."""
        try:
            return json.loads(self.body) if self.body.strip() else None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise UpstreamError(self.url, f"response is not valid JSON ({exc})") from exc


class ResponseCache:
    """SQLite cache of raw HTTP responses, keyed by URL.

    Bodies are stored zlib-compressed together with the HTTP status, content type,
    SHA-256 of the body and the time of the original download. Only definitive
    answers are cached (200, and 400/404 'does not exist'); transient failures are
    never cached, so a later run retries them.
    """

    SCHEMA_SQL = """CREATE TABLE IF NOT EXISTS responses (
        url TEXT PRIMARY KEY, status INTEGER NOT NULL, content_type TEXT, body BLOB NOT NULL,
        sha256 TEXT NOT NULL, fetched_at TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 1)"""

    def __init__(self, path: Path):
        """Open (or create) the cache database at ``path``."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(self.SCHEMA_SQL)
        self.db.commit()

    def get(self, url: str) -> Response | None:
        """Return the cached response for ``url``, or None."""
        row = self.db.execute(
            "SELECT status, content_type, body, sha256, fetched_at, attempts FROM responses WHERE url = ?", (url,)
        ).fetchone()
        if row is None:
            return None
        status, ctype, blob, digest, fetched_at, attempts = row
        return Response(url, status, zlib.decompress(blob), ctype or "", fetched_at, digest, True, attempts)

    def put(self, r: Response) -> None:
        """Store a response (replacing any previous entry for the same URL)."""
        self.db.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
            (r.url, r.status, r.content_type, zlib.compress(r.body, 9), r.sha256, r.fetched_at, r.attempts),
        )
        self.db.commit()

    def delete(self, url: str) -> None:
        """Remove one URL from the cache (used when a cached body turns out to be unusable)."""
        self.db.execute("DELETE FROM responses WHERE url = ?", (url,))
        self.db.commit()

    def urls(self, like: str = "%") -> list[str]:
        """Cached URLs matching an SQL LIKE pattern, sorted."""
        return [u for (u,) in self.db.execute("SELECT url FROM responses WHERE url LIKE ? ORDER BY url", (like,))]

    def count(self) -> int:
        """Number of cached responses."""
        return int(self.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def clear(self) -> None:
        """Delete every cached response (used by --fresh)."""
        self.db.execute("DELETE FROM responses")
        self.db.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()


class BackoffWait:
    """tenacity wait strategy: exponential backoff that honours Retry-After.

    Before retry n (1-based) it waits ``base * 2**(n-1)`` seconds (1, 2, 4, 8 s by
    default) after a timeout, connection error or HTTP 5xx. A rate limit (HTTP 429)
    waits longer, ``rate_limit_base * 2**(n-1)`` seconds (5, 10, 20, 40 s by default),
    because rate-limit windows are usually tens of seconds long; a ``Retry-After``
    header is used instead when it is longer still. Every delay is capped at ``cap``.
    """

    def __init__(self, base: float = 1.0, cap: float = 60.0, rate_limit_base: float = 5.0):
        """Set the base delays and the cap, all in seconds."""
        self.base, self.cap, self.rate_limit_base = base, cap, rate_limit_base

    def __call__(self, state: RetryCallState) -> float:
        """Return the number of seconds to sleep before the next attempt."""
        n = state.attempt_number
        delay = self.base * 2 ** (n - 1)
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, RetryableHTTPStatus) and exc.status == 429:
            delay = max(delay, self.rate_limit_base * 2 ** (n - 1))
        if isinstance(exc, RetryableHTTPStatus) and exc.retry_after:
            delay = max(delay, exc.retry_after)
        return min(delay, self.cap)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header given in seconds; HTTP-date values are ignored."""
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


def _utcnow() -> str:
    """Current UTC time as an ISO-8601 string with second precision."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SourceStats:
    """Per-source request counters and access times, for the provenance record."""

    requests: int = 0
    from_cache: int = 0
    live: int = 0
    retries: int = 0
    failed: int = 0
    status_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    first_access: str | None = None
    last_access: str | None = None
    digests: dict[str, str] = field(default_factory=dict)      # url -> sha256 of body

    def add(self, r: Response) -> None:
        """Account for one completed response."""
        self.requests += 1
        self.from_cache += r.from_cache
        self.live += not r.from_cache
        if not r.from_cache:
            self.retries += max(0, r.attempts - 1)
        self.status_counts[str(r.status)] += 1
        self.first_access = min(t for t in (self.first_access, r.fetched_at) if t)
        self.last_access = max(t for t in (self.last_access, r.fetched_at) if t)
        self.digests[r.url] = r.sha256

    def aggregate_sha256(self) -> str | None:
        """SHA-256 over the sorted (url, body digest) pairs: one fingerprint per source."""
        if not self.digests:
            return None
        lines = "\n".join(f"{u} {d}" for u, d in sorted(self.digests.items()))
        return hashlib.sha256(lines.encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready summary."""
        return {"requests": self.requests, "served_from_cache": self.from_cache, "live_requests": self.live,
                "retries": self.retries, "failed_after_retries": self.failed,
                "http_status_counts": dict(sorted(self.status_counts.items())),
                "first_access_utc": self.first_access, "last_access_utc": self.last_access,
                "responses_sha256": self.aggregate_sha256()}


class HttpFetcher:
    """Cached, polite HTTP client with tenacity retries (failure mode a).

    ``fetch(url, source)`` returns a :class:`Response` from the cache when possible,
    otherwise from the network. Network calls are retried on timeouts, connection
    errors, HTTP 429 and HTTP 5xx with :class:`BackoffWait`, at most ``max_attempts``
    times (default 5 = 1 try + 4 retries). When every attempt fails, or when
    ``offline`` is set and the URL is not cached, :class:`UpstreamError` is raised.
    """

    DEFINITIVE = (200, 400, 404)

    def __init__(self, cache: ResponseCache, *, offline: bool = False, transport: httpx.BaseTransport | None = None,
                 timeout: float = 30.0, max_attempts: int = 5, backoff_base: float = 1.0, backoff_cap: float = 60.0,
                 rate_limit_base: float = 5.0, min_interval: float = 0.25, max_interval: float = 2.0,
                 contact: str | None = None, circuit_threshold: int = 5):
        """Create the client. ``transport`` lets the tests inject an ``httpx.MockTransport``.

        Politeness: at most one request per ``min_interval`` seconds to each host. When
        a host answers HTTP 429, its interval is doubled (at least 0.5 s, at most
        ``max_interval``) for the rest of the run, so the pipeline slows down instead of
        hammering a rate-limited service.

        ``circuit_threshold``: after this many consecutive requests to one host have
        failed (each after all its retries), further requests to that host fail fast
        for the rest of the run instead of waiting through the backoff again.
        """
        if max_attempts < 4:
            raise ValueError("max_attempts must be >= 4 (at least 3 retries)")
        self.cache, self.offline = cache, offline
        self.max_attempts, self.min_interval, self.max_interval = max_attempts, min_interval, max_interval
        agent = f"{PIPELINE_NAME}/{PIPELINE_VERSION} (PHYS 7440 course project; python-httpx)"
        if contact:
            agent += f" mailto:{contact}"
        self.client = httpx.Client(transport=transport, timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
                                   headers={"User-Agent": agent}, follow_redirects=True)
        self.wait = BackoffWait(backoff_base, backoff_cap, rate_limit_base)
        self.stats: dict[str, SourceStats] = defaultdict(SourceStats)
        self._last_request: dict[str, float] = defaultdict(float)          # host -> monotonic time
        self.host_interval: dict[str, float] = defaultdict(lambda: self.min_interval)
        self.rate_limited: dict[str, int] = defaultdict(int)                # host -> number of HTTP 429 answers
        self._current_url = ""
        self.live_calls = 0
        self.circuit_threshold = circuit_threshold
        self.host_failures: dict[str, int] = defaultdict(int)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def _log_retry(self, state: RetryCallState) -> None:
        """tenacity ``before_sleep`` hook: log which request is retried, after how long, and why."""
        exc = state.outcome.exception() if state.outcome else None
        log.warning("retry %d/%d in %.1f s for %s (%s: %s)", state.attempt_number, self.max_attempts - 1,
                    state.next_action.sleep if state.next_action else 0.0, self._current_url,
                    type(exc).__name__, exc)

    def _trip(self, host: str) -> None:
        """Count one more consecutive failure for ``host``; log once when its circuit opens."""
        self.host_failures[host] += 1
        if self.host_failures[host] == self.circuit_threshold:
            log.error("%s failed %d times in a row; skipping it for the rest of this run", host, self.circuit_threshold)

    def _slow_down(self, host: str) -> None:
        """Adaptive throttle after an HTTP 429: double the request interval for ``host``."""
        self.rate_limited[host] += 1
        old = self.host_interval[host]
        new = min(self.max_interval, max(2 * old, min(0.5, self.max_interval)))
        if new > old:
            self.host_interval[host] = new
            log.warning("rate limited by %s: slowing down to one request every %.2f s", host, new)

    def _get_once(self, url: str, accept: str) -> httpx.Response:
        """One network attempt; raises RetryableHTTPStatus for HTTP 429 and 5xx."""
        host = httpx.URL(url).host
        wait = self.host_interval[host] - (time.monotonic() - self._last_request[host])
        if wait > 0:
            time.sleep(wait)
        self._last_request[host] = time.monotonic()
        self.live_calls += 1
        response = self.client.get(url, headers={"Accept": accept})
        if response.status_code == 429:
            self._slow_down(host)
        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableHTTPStatus(url, response.status_code, _parse_retry_after(response.headers.get("Retry-After")))
        return response

    def fetch(self, url: str, source: str, *, accept: str = "application/json", use_cache: bool = True) -> Response:
        """Return the response for ``url``; ``source`` labels it in the provenance statistics."""
        stats = self.stats[source]
        if use_cache:
            hit = self.cache.get(url)
            if hit is not None:
                stats.add(hit)
                return hit
        if self.offline:
            stats.failed += 1
            raise UpstreamError(url, "offline mode and not in cache")
        host = httpx.URL(url).host
        if self.host_failures[host] >= self.circuit_threshold:
            stats.failed += 1
            raise UpstreamError(url, f"{host} skipped: {self.host_failures[host]} consecutive requests failed "
                                     "after all retries in this run (circuit open); re-run later")
        self._current_url = url
        retrying = Retrying(retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS), wait=self.wait,
                            stop=stop_after_attempt(self.max_attempts), before_sleep=self._log_retry, reraise=True)
        attempts = 0
        try:
            for attempt in retrying:
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    raw = self._get_once(url, accept)
        except RETRYABLE_EXCEPTIONS as exc:
            stats.failed += 1
            stats.retries += attempts - 1
            self._trip(host)
            log.error("giving up after %d attempts: %s (%s)", attempts, url, exc)
            raise UpstreamError(url, f"failed after {attempts} attempts: {type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:                       # e.g. proxy refusal, invalid URL, TLS error: not retryable
            stats.failed += 1
            self._trip(host)
            log.error("request failed (not retryable): %s (%s: %s)", url, type(exc).__name__, exc)
            raise UpstreamError(url, f"{type(exc).__name__}: {exc}") from exc
        self.host_failures[host] = 0
        body = raw.content
        r = Response(url=url, status=raw.status_code, body=body, content_type=raw.headers.get("Content-Type", ""),
                     fetched_at=_utcnow(), sha256=hashlib.sha256(body).hexdigest(), from_cache=False, attempts=attempts)
        if use_cache and r.status in self.DEFINITIVE:
            self.cache.put(r)
        stats.add(r)
        return r


    def fetch_json(self, url: str, source: str) -> tuple[Response, Any]:
        """Like :meth:`fetch`, but also parse the body as JSON (for HTTP 200).

        A body that is not valid JSON (e.g. an HTML error page served with status 200)
        is removed from the cache before :class:`UpstreamError` is raised, so a later
        run fetches it again instead of reusing the broken copy.
        """
        r = self.fetch(url, source)
        if r.status != 200:
            return r, None
        try:
            return r, r.json()
        except UpstreamError:
            self.cache.delete(url)
            raise


# ======================================================================================
# Dataset files (S669, S2648)
# ======================================================================================


def fetch_file_source(src: FileSource, fetcher: HttpFetcher, raw_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download (or reuse) a pinned dataset file, verify its SHA-256 and parse it.

    A checksum mismatch stops the run on purpose: the test and training sets are
    frozen by the evaluation protocol, so silently using different data would
    invalidate every comparison with the Week 8 baseline.
    """
    r = fetcher.fetch(src.url, src.name, accept="text/csv, text/plain, */*")
    if r.status != 200:
        raise UpstreamError(src.url, f"HTTP {r.status} for dataset file {src.name}")
    if r.sha256 != src.sha256:
        raise RuntimeError(f"{src.name}: SHA-256 {r.sha256} differs from the frozen {src.sha256}; "
                           f"the source changed. Refusing to continue ({src.url}).")
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / src.local_name).write_bytes(r.body)
    df = pd.read_csv(io.BytesIO(r.body), dtype=str, keep_default_na=False)
    meta = {"access_timestamp_utc": r.fetched_at, "sha256": r.sha256, "bytes": len(r.body), "rows": len(df),
            "served_from_cache": r.from_cache, "local_copy": f"data/raw/{src.local_name}"}
    return df, meta


def upstream_latest_commit(src: FileSource, fetcher: HttpFetcher) -> dict[str, str]:
    """Ask the GitHub API for the newest commit touching the dataset file (provenance only).

    Never cached and never fatal: on any failure the fields are 'MISSING' and the
    reason is logged.
    """
    url = GITHUB_COMMITS.format(repo=src.repo, path=src.path)
    try:
        r = fetcher.fetch(url, "GitHub commits API", accept="application/vnd.github+json", use_cache=False)
        payload = r.json()
        if r.status != 200 or not isinstance(payload, list) or not payload:
            raise UpstreamError(url, f"HTTP {r.status} or empty commit list")
        sha = payload[0].get("sha") or SENTINEL_STR
        date = ((payload[0].get("commit") or {}).get("committer") or {}).get("date") or SENTINEL_STR
        return {"sha": sha, "date": date, "checked_at_utc": r.fetched_at,
                "matches_pinned_file_commit": sha == src.file_commit}
    except UpstreamError as exc:
        log.warning("could not read the latest upstream commit of %s: %s", src.name, exc.reason)
        return {"sha": SENTINEL_STR, "date": SENTINEL_STR, "checked_at_utc": _utcnow(), "error": exc.reason}


def parse_mutation(text: str) -> tuple[str, int, str, str] | None:
    """'S11A' -> ('S', 11, '', 'A'); 'L27CN' (insertion code C) -> ('L', 27, 'C', 'N'); None if invalid."""
    m = MUTATION_RE.match(str(text).strip().upper())
    if not m or m.group(1) not in AA20 or m.group(4) not in AA20:
        return None
    return m.group(1), int(m.group(2)), m.group(3), m.group(4)


def _to_float(value: Any) -> float | None:
    """Convert a CSV cell to float; None for blank, NA or non-numeric cells."""
    try:
        x = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _empty_row() -> dict[str, Any]:
    """A row with every data column set to its sentinel and neutral status columns."""
    row = {c: sentinel_for(c) for c in COLUMNS}
    row.update(numbering_map="unmapped", wt_check="unverified", missing_fields="")
    return row


def _is_sentinel(column: str, value: Any) -> bool:
    """True if ``value`` is the documented sentinel of ``column``."""
    if column == "ddg_kcal_mol":
        return isinstance(value, float) and math.isnan(value)
    return value == sentinel_for(column)


def _finish_missing(row: dict[str, Any]) -> dict[str, Any]:
    """Fill 'missing_fields' with every data column that still holds its sentinel."""
    row["missing_fields"] = ";".join(c for c in COLUMNS if c not in STATUS_COLUMNS and _is_sentinel(c, row[c]))
    return row


def normalise_s669(df: pd.DataFrame, src: FileSource, mlog: MissingFieldLog) -> list[dict[str, Any]]:
    """Map S669 rows onto the internal schema (already in UniProt numbering).

    ID cleaning, as in Assignment 8: 19 IDs carry the mutation as a suffix
    ('P04517WTE2143G' -> 'P04517'); 3 PDB-only chains have no UniProt accession and
    get the 'MISSING' sentinel. The wild type is checked against the file's own
    sequence column. pH and temperature are not part of this file and get sentinels.
    """
    rows = []
    mlog.record("*", ["ph", "temperature_k"], src.url, "column not provided by the S669 file (all rows)")
    factor = DDG_UNIT_FACTORS[src.ddg_unit] * src.ddg_sign
    for i, rec in enumerate(df.to_dict("records")):
        row = _empty_row()
        vid = f"{src.name}:{i:04d}"
        raw_id = str(rec.get("ID", "")).strip()
        protein = re.sub(r"WT[A-Z]\d+[A-Z]$", "", raw_id)
        row.update(variant_id=vid, dataset=src.name, split=src.role, source_row=i, source_id=raw_id or SENTINEL_STR,
                   source_mutation=str(rec.get("MTS", "")).strip() or SENTINEL_STR,
                   group=protein or SENTINEL_STR, protein=protein or SENTINEL_STR,
                   numbering_map="source (UniProt numbering)", _source_url=src.url)
        has_accession = bool(UNIPROT_RE.match(protein))
        if has_accession:
            row["uniprot_accession"] = protein
        else:
            row["numbering_map"] = "unmapped (PDB chain without UniProt accession)"
            mlog.record(vid, ["uniprot_accession", "position_uniprot", "mutation_uniprot"], src.url,
                        f"source ID {raw_id!r} is not a UniProt accession (PDB chain), so there is no UniProt numbering")
        parsed = parse_mutation(row["source_mutation"])
        if parsed is None:
            mlog.record(vid, ["wt", "mt", "position_uniprot", "mutation_uniprot"], src.url,
                        f"cannot parse mutation {row['source_mutation']!r}")
        else:
            wt, pos, _icode, mt = parsed
            row.update(wt=wt, mt=mt)
            if has_accession:
                row.update(position_uniprot=pos, mutation_uniprot=f"{wt}{pos}{mt}")
            seq = str(rec.get("Sequence", "")).strip().upper()
            row["wt_check"] = ("match" if 0 < pos <= len(seq) and seq[pos - 1] == wt
                               else "mismatch" if seq else "unverified")
        ddg = _to_float(rec.get("DDG"))
        if ddg is None:
            mlog.record(vid, ["ddg_kcal_mol"], src.url, "ΔΔG is blank or not numeric")
        else:
            row["ddg_kcal_mol"] = round(ddg * factor, 4)
        rows.append(row)
    return rows


def normalise_s2648(df: pd.DataFrame, src: FileSource, mlog: MissingFieldLog) -> list[dict[str, Any]]:
    """Map S2648 rows onto the internal schema, still in PDB numbering.

    Converts the temperature from °C to K and keeps the PDB chain and author residue
    number in private ``_pdb``/``_chain``/``_resnum``/``_icode`` keys; the UniProt
    accession and position are filled in later by :func:`map_pdb_to_uniprot`.
    """
    rows = []
    factor = DDG_UNIT_FACTORS[src.ddg_unit] * src.ddg_sign
    for i, rec in enumerate(df.to_dict("records")):
        row = _empty_row()
        vid = f"{src.name}:{i:04d}"
        pdb, chain = str(rec.get("PDB", "")).strip().upper(), str(rec.get("CHAIN", "")).strip()
        row.update(variant_id=vid, dataset=src.name, split=src.role, source_row=i,
                   source_id=f"{pdb}_{chain}" if pdb else SENTINEL_STR,
                   source_mutation=str(rec.get("MUT", "")).strip() or SENTINEL_STR,
                   group=str(rec.get("CLID", "")).strip() or SENTINEL_STR,
                   protein=f"{pdb}_{chain}" if pdb else SENTINEL_STR, _source_url=src.url,
                   _pdb=pdb, _chain=chain, _resnum=None, _icode="")
        parsed = parse_mutation(row["source_mutation"])
        if parsed is None:
            mlog.record(vid, ["wt", "mt"], src.url, f"cannot parse mutation {row['source_mutation']!r}")
        else:
            wt, num, icode, mt = parsed
            row.update(wt=wt, mt=mt, _resnum=num, _icode=icode)
        ddg = _to_float(rec.get("DDG"))
        if ddg is None:
            mlog.record(vid, ["ddg_kcal_mol"], src.url, "ΔΔG is blank or not numeric")
        else:
            row["ddg_kcal_mol"] = round(ddg * factor, 4)
        ph = _to_float(rec.get("pH"))
        if ph is None:
            mlog.record(vid, ["ph"], src.url, "pH is blank or not numeric")
        else:
            row["ph"] = ph
        t = _to_float(rec.get("T"))
        if t is None:
            mlog.record(vid, ["temperature_k"], src.url, "temperature is blank or not numeric")
        else:
            row["temperature_k"] = round(t + 273.15, 2) if src.temperature_unit == "degC" else t
        rows.append(row)
    return rows


def check_sign_convention(rows: list[dict[str, Any]], name: str) -> float:
    """Fraction of variants with ΔΔG < 0.

    Most measured mutations destabilise, so a fraction above 0.6 confirms the
    'negative = destabilising' convention; anything else stops the run.
    """
    ddg = np.array([r["ddg_kcal_mol"] for r in rows], float)
    ddg = ddg[~np.isnan(ddg)]
    frac = float((ddg < 0).mean()) if len(ddg) else float("nan")
    if not frac > 0.6:
        raise RuntimeError(f"{name}: only {frac:.0%} of ΔΔG values are negative; the sign convention looks inverted")
    return frac


# ======================================================================================
# Identifier normalisation: PDB numbering -> UniProt numbering (SIFTS)
# ======================================================================================


@dataclass(frozen=True)
class SiftsSegment:
    """One SIFTS segment: a run of PDB author residues that maps linearly onto UniProt."""

    accession: str
    chain: str
    author_start: int | None
    author_end: int | None
    unp_start: int
    unp_end: int

    @property
    def offset(self) -> int | None:
        """UniProt number minus PDB author number inside this segment."""
        return None if self.author_start is None else self.unp_start - self.author_start

    def covers(self, resnum: int) -> bool:
        """True if the author residue number lies inside the segment."""
        return (self.author_start is not None and self.author_end is not None
                and self.author_start <= resnum <= self.author_end)


def parse_sifts(payload: Any, pdb_id: str, chain: str) -> list[SiftsSegment]:
    """Extract the UniProt segments of one PDB chain from a PDBe SIFTS response.

    Matches the author chain ID first and falls back to the label (struct_asym) ID.
    Returns an empty list when the chain has no UniProt mapping or the payload is
    malformed.
    """
    if not isinstance(payload, dict):
        return []
    entry = payload.get(pdb_id.lower()) or payload.get(pdb_id.upper()) or {}
    unp = entry.get("UniProt") if isinstance(entry, dict) else None
    if not isinstance(unp, dict):
        return []
    by_author, by_label = [], []
    for acc, info in unp.items():
        for m in (info or {}).get("mappings", []) or []:
            start, end = m.get("start") or {}, m.get("end") or {}
            if m.get("unp_start") is None or m.get("unp_end") is None:
                continue
            a_start, a_end = start.get("author_residue_number"), end.get("author_residue_number")
            if a_end is None and a_start is not None:
                # SIFTS leaves the author number null when the last residue is unobserved; a segment is a
                # gap-free run, so its author span equals its UniProt span (checked later against the sequence)
                a_end = a_start + int(m["unp_end"]) - int(m["unp_start"])
            seg = SiftsSegment(accession=str(acc), chain=str(m.get("chain_id")), author_start=a_start,
                               author_end=a_end, unp_start=int(m["unp_start"]), unp_end=int(m["unp_end"]))
            if m.get("chain_id") == chain:
                by_author.append(seg)
            elif m.get("struct_asym_id") == chain:
                by_label.append(seg)
    return sorted(by_author or by_label, key=lambda s: (s.accession, s.unp_start))


def fetch_sifts(pdb_ids: Iterable[str], fetcher: HttpFetcher) -> tuple[dict[str, Any], dict[str, str]]:
    """SIFTS UniProt mappings for each PDB ID: {pdb: payload or None}, {pdb: URL used}.

    Tries the PDBe REST API first and the PDBe graph API (same JSON format) when the
    first answer is not a usable mapping.
    """
    payloads, used = {}, {}
    for pdb in pdb_ids:
        payloads[pdb], used[pdb] = None, PDBE_SIFTS.format(pdb_id=pdb.lower())
        for template in (PDBE_SIFTS, PDBE_SIFTS_FALLBACK):
            url = template.format(pdb_id=pdb.lower())
            try:
                _, payload = fetcher.fetch_json(url, "PDBe SIFTS mappings")
            except UpstreamError as exc:
                log.debug("SIFTS lookup failed for %s: %s", pdb, exc.reason)
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get(pdb.lower()) or payload.get(pdb.upper()), dict):
                payloads[pdb], used[pdb] = payload, url
                break
            log.debug("no usable SIFTS mapping for %s at %s", pdb, url)
    return payloads, used


def best_offset(sites: Iterable[tuple[str, int]], sequence: str,
                candidates: Iterable[int] | None = None) -> tuple[int | None, float]:
    """Offset k that makes the most wild-type residues match ``sequence[n + k - 1]``.

    ``sites`` are distinct (wild type, author number) pairs of one chain. With
    ``candidates`` only those offsets are tried (e.g. the SIFTS segment offsets);
    otherwise every offset that keeps all sites inside the sequence is tried.
    Returns (offset, fraction matched); offset is None if the best offset is not
    unique.
    """
    sites = sorted(set(sites))
    if not sites or not sequence:
        return None, 0.0
    lo, hi = min(n for _, n in sites), max(n for _, n in sites)
    pool = sorted(set(candidates)) if candidates is not None else range(1 - lo, len(sequence) - hi + 1)
    scores = []
    for k in pool:
        hits = sum(1 for wt, n in sites if 0 < n + k <= len(sequence) and sequence[n + k - 1] == wt)
        scores.append((hits, k))
    if not scores:
        return None, 0.0
    scores.sort(key=lambda t: (-t[0], abs(t[1]), t[1]))
    best_hits, best_k = scores[0]
    if len(scores) > 1 and scores[1][0] == best_hits:
        return None, best_hits / len(sites)
    return best_k, best_hits / len(sites)


def map_pdb_to_uniprot(rows: list[dict[str, Any]], segments: dict[str, list[SiftsSegment]],
                       sequences: dict[str, str], sifts_urls: dict[str, str], mlog: MissingFieldLog) -> None:
    """Fill ``uniprot_accession`` and ``position_uniprot`` of S2648 rows in place.

    For each variant: (1) take the SIFTS segment that covers the author residue and
    apply its offset; (2) check that the wild type matches the UniProt sequence; (3)
    if not (or for residues outside every segment), use the chain-wide offset that
    explains most wild types (:func:`best_offset`; a free search needs >= 3 distinct
    sites and >= 80 % matches). Residues with PDB insertion codes, or that still do
    not match, keep the sentinel and are logged.
    """
    by_chain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chain[row["protein"]].append(row)
    for key, chain_rows in by_chain.items():
        segs = segments.get(key, [])
        url = sifts_urls.get(key, PDBE_SIFTS.format(pdb_id=key.split("_")[0].lower()))
        if not segs:
            for row in chain_rows:
                mlog.record(row["variant_id"], ["uniprot_accession", "position_uniprot", "mutation_uniprot"], url,
                            f"no SIFTS UniProt mapping for PDB chain {key}")
            continue
        sites = {(r["wt"], r["_resnum"]) for r in chain_rows if r["_resnum"] is not None and not r["_icode"]}
        votes: dict[str, int | None] = {}
        for row in chain_rows:
            vid, n = row["variant_id"], row["_resnum"]
            if n is None:
                continue
            seg = next((s for s in segs if s.covers(n)), None)
            accession = (seg or segs[0]).accession
            seq = sequences.get(accession, "")
            row["uniprot_accession"] = accession
            if row["_icode"]:
                mlog.record(vid, ["position_uniprot", "mutation_uniprot"], url,
                            f"PDB residue {n}{row['_icode']} has an insertion code; no linear UniProt mapping")
                continue
            pos, method = None, None
            if seg is not None and seg.offset is not None:
                pos, method = n + seg.offset, "SIFTS segment offset"
            if seq and (pos is None or not (0 < pos <= len(seq) and seq[pos - 1] == row["wt"])):
                if accession not in votes:
                    seg_offsets = [s.offset for s in segs if s.accession == accession and s.offset is not None]
                    k, frac = best_offset(sites, seq, seg_offsets) if seg_offsets else (None, 0.0)
                    if (k is None or frac < OFFSET_VOTE_MIN_FRACTION) and len(sites) >= OFFSET_VOTE_MIN_SITES:
                        k, frac = best_offset(sites, seq)
                    votes[accession] = k if (k is not None and frac >= OFFSET_VOTE_MIN_FRACTION) else None
                k = votes[accession]
                if k is not None and 0 < n + k <= len(seq) and seq[n + k - 1] == row["wt"]:
                    pos, method = n + k, "SIFTS accession + chain-wide offset vote"
            if pos is None or pos < 1:
                mlog.record(vid, ["position_uniprot", "mutation_uniprot"], url,
                            f"residue {n} of {key} lies outside every SIFTS segment and no offset fits")
                continue
            row.update(position_uniprot=pos, mutation_uniprot=f"{row['wt']}{pos}{row['mt']}", numbering_map=method)
            if not seq:
                row["wt_check"] = "unverified"
            elif 0 < pos <= len(seq) and seq[pos - 1] == row["wt"]:
                row["wt_check"] = "match"
            else:
                row["wt_check"] = "mismatch"
                mlog.record(vid, ["wt_check"], url, f"wild type {row['wt']} not found at UniProt {accession} position {pos}")


# ======================================================================================
# AlphaFold DB, AlphaMissense and structure-derived features
# ======================================================================================


def choose_afdb_record(payload: Any, accession: str) -> dict[str, Any] | None:
    """Pick the canonical AlphaFold DB record for an accession (same rule as Assignment 7):
    exact accession, GDM provider, fragment F1, monomer, newest version."""
    records = [r for r in payload if isinstance(r, dict)] if isinstance(payload, list) else []
    if not records:
        return None
    exact = [r for r in records if r.get("uniprotAccession") == accession]

    def rank(r: dict[str, Any]) -> tuple:
        """Sort key: preferred records first."""
        return (r.get("providerId") != "GDM", not str(r.get("entryId", "")).endswith("-F1"),
                bool(r.get("isComplex")), -(r.get("latestVersion") or 0))

    return sorted(exact or records, key=rank)[0]


def afdb_file_url(record: dict[str, Any], accession: str, kind: str) -> str | None:
    """URL of an AlphaFold DB file of the pinned model version.

    ``kind`` is 'confidence', 'model' or 'aa-substitutions'. The record's own URL
    is used when its version equals the pinned version; otherwise the pinned-version
    URL is built from the entry ID. The AlphaMissense URL is returned as given
    (None for non-human proteins).
    """
    if kind == "aa-substitutions":
        return record.get("amAnnotationsUrl") or None
    key, suffix = {"confidence": ("plddtDocUrl", "confidence"), "model": ("pdbUrl", "model")}[kind]
    ext = "json" if kind == "confidence" else "pdb"
    entry = record.get("entryId") or f"AF-{accession}-F1"
    url = record.get(key)
    if url and record.get("latestVersion") == AFDB_MODEL_VERSION:
        return url
    return f"{AFDB_BASE}/files/{entry}-{suffix}_v{AFDB_MODEL_VERSION}.{ext}"


def parse_confidence(payload: Any) -> dict[int, float]:
    """Per-residue pLDDT from an AlphaFold confidence JSON: {model residue number: pLDDT}.

    Null or non-numeric entries are skipped (they become missing fields later).
    """
    if isinstance(payload, list) and payload:          # tolerate a list-wrapped object
        payload = payload[0]
    if not isinstance(payload, dict):
        return {}
    nums, scores = payload.get("residueNumber") or [], payload.get("confidenceScore") or []
    out = {}
    for n, s in zip(nums, scores):
        try:
            if n is not None and s is not None:
                out[int(n)] = float(s)
        except (TypeError, ValueError):
            continue
    return out


def parse_alphamissense(text: str) -> dict[str, tuple[float, str]]:
    """AlphaMissense table -> {'E7V': (0.2229, 'likely_benign'), ...}; header and bad lines skipped."""
    out = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or parts[0] == "protein_variant":
            continue
        score = _to_float(parts[1])
        if score is None:
            continue
        out[parts[0]] = (score, AM_CLASS.get(parts[2] if len(parts) > 2 else "", SENTINEL_STR))
    return out


def parse_pdb_side_chain_centres(text: str) -> dict[int, np.ndarray]:
    """{residue number: Cβ coordinates (Å)} from a PDB file, Cα for glycine; first model only."""
    cb: dict[int, np.ndarray] = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        name, resn = line[12:16].strip(), line[17:20].strip()
        try:
            num = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        if name == "CB" or (name == "CA" and resn == "GLY"):
            cb[num] = xyz
    return cb


def contact_numbers(centres: dict[int, np.ndarray], radius: float = BURIAL_RADIUS_A) -> dict[int, int]:
    """Number of other residues whose Cβ lies within ``radius`` Å of each residue's Cβ.

    A simple, parameter-light burial measure: surface residues have few neighbours,
    core residues many (very roughly < 15 versus > 25 at 10 Å).
    """
    if not centres:
        return {}
    nums = np.array(sorted(centres))
    xyz = np.stack([centres[n] for n in nums])
    counts = np.empty(len(nums), dtype=int)
    for i in range(0, len(nums), 256):                    # chunks keep memory small for long proteins
        d2 = ((xyz[i:i + 256, None, :] - xyz[None, :, :]) ** 2).sum(-1)
        counts[i:i + 256] = (d2 <= radius ** 2).sum(1) - 1
    return {int(n): int(c) for n, c in zip(nums, counts)}


def parse_fasta(text: str) -> str:
    """Sequence of the first record of a FASTA text ('' if there is none)."""
    seq, started = [], False
    for line in text.splitlines():
        if line.startswith(">"):
            if started:
                break
            started = True
            continue
        if started:
            seq.append(line.strip())
    return "".join(seq).upper()


@dataclass
class ProteinData:
    """Everything fetched for one UniProt accession."""

    accession: str
    prediction_url: str
    record: dict[str, Any] | None = None
    record_error: str | None = None
    sequence: str = ""
    sequence_source: str | None = None
    plddt: dict[int, float] | None = None
    plddt_url: str | None = None
    plddt_error: str | None = None
    burial: dict[int, int] | None = None
    model_url: str | None = None
    model_error: str | None = None
    am: dict[str, tuple[float, str]] | None = None
    am_url: str | None = None
    am_error: str | None = None


def fetch_prediction(acc: str, fetcher: HttpFetcher) -> ProteinData:
    """AlphaFold DB prediction record for one accession (HTTP 404 = no model; logged later)."""
    pdata = ProteinData(accession=acc, prediction_url=AFDB_PREDICTION.format(accession=acc))
    try:
        r, payload = fetcher.fetch_json(pdata.prediction_url, "AlphaFold DB prediction API")
        if r.status == 200:
            pdata.record = choose_afdb_record(payload, acc)
            if pdata.record is None:
                pdata.record_error = "AlphaFold DB returned an empty record list"
        else:
            pdata.record_error = f"AlphaFold DB has no model for {acc} (HTTP {r.status})"
    except UpstreamError as exc:
        pdata.record_error = exc.reason
    if pdata.record and pdata.record.get("uniprotSequence"):
        pdata.sequence, pdata.sequence_source = str(pdata.record["uniprotSequence"]).upper(), "AlphaFold DB uniprotSequence"
    return pdata


def fetch_uniprot_sequence(pdata: ProteinData, fetcher: HttpFetcher) -> None:
    """Fallback reference sequence from UniProt REST, used only to validate numbering."""
    url = UNIPROT_FASTA.format(accession=pdata.accession)
    try:
        r = fetcher.fetch(url, "UniProt REST (FASTA)", accept="text/plain")
        if r.status == 200:
            seq = parse_fasta(r.text())
            if seq:
                pdata.sequence, pdata.sequence_source = seq, "UniProt REST FASTA"
    except UpstreamError as exc:
        log.debug("UniProt sequence for %s unavailable: %s", pdata.accession, exc.reason)


def fetch_structure_files(pdata: ProteinData, fetcher: HttpFetcher) -> None:
    """Download pLDDT, coordinates and (human proteins only) AlphaMissense for one record."""
    rec, acc = pdata.record, pdata.accession
    if rec is None:
        return
    pdata.plddt_url = afdb_file_url(rec, acc, "confidence")
    try:
        r, payload = fetcher.fetch_json(pdata.plddt_url, "AlphaFold DB confidence (pLDDT) files")
        if r.status == 200:
            pdata.plddt = parse_confidence(payload)
        else:
            pdata.plddt_error = f"HTTP {r.status}"
    except UpstreamError as exc:
        pdata.plddt_error = exc.reason
    pdata.model_url = afdb_file_url(rec, acc, "model")
    try:
        r = fetcher.fetch(pdata.model_url, "AlphaFold DB model coordinates (PDB)", accept="chemical/x-pdb, text/plain, */*")
        if r.status == 200:
            pdata.burial = contact_numbers(parse_pdb_side_chain_centres(r.text()))
        else:
            pdata.model_error = f"HTTP {r.status}"
    except UpstreamError as exc:
        pdata.model_error = exc.reason
    pdata.am_url = afdb_file_url(rec, acc, "aa-substitutions")
    if not pdata.am_url:
        pdata.am_error = "no AlphaMissense table in AlphaFold DB (AlphaMissense covers human proteins only)"
        return
    try:
        r = fetcher.fetch(pdata.am_url, "AlphaMissense tables (AlphaFold DB)", accept="text/csv, */*")
        if r.status == 200:
            pdata.am = parse_alphamissense(r.text())
        else:
            pdata.am_error = f"HTTP {r.status}"
    except UpstreamError as exc:
        pdata.am_error = exc.reason


def add_structure_features(row: dict[str, Any], pdata: ProteinData | None, mlog: MissingFieldLog) -> None:
    """Fill the AlphaFold / AlphaMissense columns of one variant row in place.

    Any value that cannot be obtained keeps its sentinel and is logged with the URL
    of the response it should have come from.
    """
    vid = row["variant_id"]
    if row["uniprot_accession"] == SENTINEL_STR or pdata is None:
        mlog.record(vid, AF_FIELDS, row.get("_source_url", ""), "no UniProt accession, so no AlphaFold DB lookup")
        return
    if pdata.record is None:
        mlog.record(vid, AF_FIELDS, pdata.prediction_url, pdata.record_error or "no AlphaFold DB record")
        return
    if row["position_uniprot"] == SENTINEL_INT or row["wt_check"] == "mismatch":
        mlog.record(vid, AF_FIELDS, pdata.prediction_url, "residue position not mapped, or wild type does not match")
        return
    rec, pos = pdata.record, int(row["position_uniprot"])
    model_seq = str(rec.get("sequence") or rec.get("uniprotSequence") or "").upper()
    start = int(rec.get("uniprotStart") or rec.get("sequenceStart") or 1)
    model_pos = pos - start + 1
    if not (0 < model_pos <= len(model_seq)) or model_seq[model_pos - 1] != row["wt"]:
        mlog.record(vid, AF_FIELDS, pdata.prediction_url,
                    f"position {pos} is outside the AlphaFold model or the model residue differs")
        return
    row["af_entry_id"] = rec.get("entryId") or SENTINEL_STR
    row["af_model_version"] = int(rec["latestVersion"]) if rec.get("latestVersion") is not None else SENTINEL_INT
    row["af_model_created"] = rec.get("modelCreatedDate") or SENTINEL_STR
    for f in ("af_entry_id", "af_model_version", "af_model_created"):
        if _is_sentinel(f, row[f]):
            mlog.record(vid, [f], pdata.prediction_url, "field is null in the AlphaFold DB record")
    row["rel_position"] = round(pos / len(pdata.sequence or model_seq), 4)
    # pLDDT (model numbering)
    if pdata.plddt is None:
        mlog.record(vid, ["plddt", "plddt_window"], pdata.plddt_url or pdata.prediction_url,
                    pdata.plddt_error or "no pLDDT file")
    elif model_pos not in pdata.plddt:
        mlog.record(vid, ["plddt", "plddt_window"], pdata.plddt_url, f"pLDDT is null or absent for model residue {model_pos}")
    else:
        row["plddt"] = round(pdata.plddt[model_pos], 2)
        window = [pdata.plddt[k] for k in range(model_pos - PLDDT_WINDOW, model_pos + PLDDT_WINDOW + 1) if k in pdata.plddt]
        row["plddt_window"] = round(float(np.mean(window)), 3)
    # burial (model numbering)
    if pdata.burial is None:
        mlog.record(vid, ["burial_cn10"], pdata.model_url or pdata.prediction_url, pdata.model_error or "no coordinates")
    elif model_pos not in pdata.burial:
        mlog.record(vid, ["burial_cn10"], pdata.model_url, f"no Cβ/Cα atom for model residue {model_pos}")
    else:
        row["burial_cn10"] = pdata.burial[model_pos]
    # AlphaMissense (UniProt numbering)
    if pdata.am is None:
        mlog.record(vid, AM_FIELDS, pdata.am_url or pdata.prediction_url, pdata.am_error or "no AlphaMissense table")
        return
    key = f"{row['wt']}{pos}{row['mt']}"
    if key in pdata.am:
        row["am_pathogenicity"], row["am_class"] = round(pdata.am[key][0], 4), pdata.am[key][1]
        if row["am_class"] == SENTINEL_STR:
            mlog.record(vid, ["am_class"], pdata.am_url, "AlphaMissense class is blank or unknown")
    else:
        mlog.record(vid, ["am_pathogenicity", "am_class"], pdata.am_url, f"{key} is not in the AlphaMissense table")
    site = [v[0] for k, v in pdata.am.items() if k[:-1] == f"{row['wt']}{pos}"]
    if site:
        row["am_position_mean"] = round(float(np.mean(site)), 4)
    else:
        mlog.record(vid, ["am_position_mean"], pdata.am_url, f"no AlphaMissense scores at {row['wt']}{pos}")


# ======================================================================================
# Outputs, hashing and provenance
# ======================================================================================


def to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows -> DataFrame with exactly the schema columns and dtypes; S669 first, then source order."""
    df = pd.DataFrame([{c: r[c] for c in COLUMNS} for r in rows], columns=COLUMNS)
    for c, (kind, _, _) in SCHEMA.items():
        df[c] = df[c].astype({"int": "int64", "float": "float64", "str": str}[kind])
    order = df["dataset"].map({S669.name: 0, S2648.name: 1}).fillna(2)
    df = df.assign(_order=order).sort_values(["_order", "source_row"], kind="mergesort")
    return df.drop(columns="_order").reset_index(drop=True)


def content_sha256(df: pd.DataFrame) -> str:
    """SHA-256 of a canonical CSV serialisation (fixed column order, row order and float format).

    Independent of the Parquet library version, so two runs that produce the same
    data produce the same hash.
    """
    buf = io.StringIO()
    df[COLUMNS].to_csv(buf, index=False, float_format="%.10g", lineterminator="\n", na_rep="NaN")
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()


def diff_tables(old: pd.DataFrame | None, new: pd.DataFrame) -> dict[str, Any] | None:
    """Row- and column-level differences between the previous and the new normalised table.

    Used by the reproducibility check: if the content hash changes, this says where
    (e.g. 'plddt: 12 rows'), so a reviewer can tell an upstream data change from a bug.
    Rows are matched on ``variant_id``. Returns None when there is no previous table.
    """
    if old is None:
        return None
    o, n = old.set_index("variant_id"), new.set_index("variant_id")
    common = o.index.intersection(n.index)
    changed = {}
    for c in COLUMNS:
        if c == "variant_id" or c not in o.columns:
            continue
        a, b = o.loc[common, c], n.loc[common, c]
        k = int((~((a == b) | (a.isna() & b.isna()))).sum())      # NaN == NaN counts as unchanged
        if k:
            changed[c] = k
    return {"rows_only_in_previous": int(len(o.index.difference(n.index))),
            "rows_only_in_new": int(len(n.index.difference(o.index))),
            "changed_values_by_column": changed}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package_versions() -> dict[str, str]:
    """Versions of Python and of the packages that can influence the outputs."""
    out = {"python": platform.python_version()}
    for pkg in ("httpx", "tenacity", "numpy", "pandas", "pyarrow"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = SENTINEL_STR
    return out


@dataclass
class Paths:
    """All file locations used by one pipeline run, relative to the project folder."""

    root: Path

    @property
    def raw(self) -> Path:
        """Verbatim copies of the dataset files."""
        return self.root / "data" / "raw"

    @property
    def cache(self) -> Path:
        """SQLite cache of raw HTTP responses."""
        return self.root / "data" / "cache" / "http_cache_ly.sqlite"

    @property
    def processed(self) -> Path:
        """Folder of the normalised outputs."""
        return self.root / "data" / "processed"

    @property
    def parquet(self) -> Path:
        """Main output: the normalised variant table (Parquet)."""
        return self.processed / "variants_ly.parquet"

    @property
    def csv(self) -> Path:
        """The same table as CSV, for reading and diffing."""
        return self.processed / "variants_ly.csv"

    @property
    def missing(self) -> Path:
        """Every substituted field with its source URL."""
        return self.processed / "missing_fields_ly.csv"

    @property
    def provenance(self) -> Path:
        """The data provenance record."""
        return self.root / "provenance_ly.json"

    @property
    def logfile(self) -> Path:
        """Run log, including one line per missing field and every retry."""
        return self.root / "logs" / "pipeline_ly.log"


def setup_logging(logfile: Path, verbose: bool = False) -> None:
    """Log everything to ``logfile`` and progress, retries and errors to the console."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("af_ddg")
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.addFilter(lambda rec: verbose or rec.name != "af_ddg.missing")
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    root.addHandler(fh)
    root.addHandler(ch)


def _rel(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` when possible, as a POSIX string."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_provenance(*, paths: Paths, df: pd.DataFrame, file_meta: dict[str, dict[str, Any]],
                     upstream: dict[str, dict[str, Any]], fetcher: HttpFetcher, proteins: dict[str, ProteinData],
                     mlog: MissingFieldLog, sign_fractions: dict[str, float], started: str, argv: list[str],
                     previous: dict[str, Any] | None, diff: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the provenance record: one entry per dataset or service, plus outputs and schema."""
    st = fetcher.stats
    models = sorted(({"accession": a, "entry_id": p.record.get("entryId"), "latest_version": p.record.get("latestVersion"),
                      "model_created": p.record.get("modelCreatedDate"), "sequence_checksum": p.record.get("sequenceChecksum"),
                      "has_alphamissense": bool(p.record.get("amAnnotationsUrl"))}
                     for a, p in proteins.items() if p.record), key=lambda m: m["accession"])
    no_model = sorted(a for a, p in proteins.items() if p.record is None)
    versions_seen = sorted({str(m["latest_version"]) for m in models})
    content = content_sha256(df)
    datasets = []
    for src in FILE_SOURCES:
        meta = file_meta[src.name]
        n = int((df["dataset"] == src.name).sum())
        tr = [f"downloaded from the commit-pinned raw URL and verified against SHA-256 {src.sha256}",
              f"ΔΔG unit {src.ddg_unit} -> kcal/mol (factor {DDG_UNIT_FACTORS[src.ddg_unit]:g}); sign checked: "
              f"{sign_fractions[src.name]:.1%} of variants have ΔΔG < 0, i.e. negative = destabilising, no flip applied"]
        if src.name == "S669":
            tr += ["IDs cleaned: mutation suffix removed ('P04517WTE2143G' -> 'P04517'); PDB-only chains get "
                   "uniprot_accession = MISSING", "wild type checked against the file's UniProt sequence column and "
                   "against the current UniProt sequence from AlphaFold DB", "pH and temperature are not in this file -> sentinel -1"]
        else:
            tr += ["temperature converted from °C to K (+273.15)",
                   "PDB author numbering mapped to UniProt numbering with PDBe SIFTS segments, validated against the "
                   "UniProt sequence, with a chain-wide offset vote as fallback (column numbering_map)"]
        tr += [f"joined with AlphaFold DB v{AFDB_MODEL_VERSION} features on (UniProt accession, UniProt position)",
               f"{n} rows written to {_rel(paths.parquet, paths.root)}"]
        datasets.append({"name": src.name, "role": src.role, "type": "file", "source_url": src.url,
                         "repository": f"https://github.com/{src.repo}", "commit_pinned_in_url": src.commit,
                         "file_last_commit": src.file_commit, "file_last_commit_date": src.file_commit_date,
                         "upstream_latest_commit": upstream.get(src.name), "doi": src.doi, "citation": src.citation,
                         "original_numbering": src.numbering, **meta, "transformations": tr})
    services = [
        ("PDBe SIFTS mappings", "identifier normalisation of S2648 (PDB chain -> UniProt accession and numbering)",
         f"{PDBE_SIFTS} (fallback: {PDBE_SIFTS_FALLBACK})", "SIFTS through the live PDBe API (no release number exposed); fingerprinted by responses_sha256",
         ["chain selected by author chain ID (fallback: label asym ID)",
          "segment offset unp_start - author_start applied to the author residue number",
          f"wild type validated against the UniProt sequence; chain-wide offset vote when >= {OFFSET_VOTE_MIN_SITES} "
          f"sites and >= {OFFSET_VOTE_MIN_FRACTION:.0%} agree", "insertion-code residues left unmapped (sentinel)"]),
        ("AlphaFold DB prediction API", "model metadata and reference UniProt sequence", AFDB_PREDICTION,
         f"AlphaFold DB; model versions seen: {', '.join(versions_seen) or 'none'}; files pinned to v{AFDB_MODEL_VERSION}",
         ["record chosen as in Assignment 7: exact accession, GDM provider, F1 fragment, monomer, newest version",
          "uniprotSequence used to validate residue numbering"]),
        ("AlphaFold DB confidence (pLDDT) files", "per-residue model confidence",
         AFDB_FILE.format(accession="{accession}", kind=f"confidence_v{AFDB_MODEL_VERSION}.json"),
         f"AlphaFold DB model v{AFDB_MODEL_VERSION}",
         ["pLDDT at the site (2 dp)", f"mean pLDDT over ±{PLDDT_WINDOW} residues (3 dp)"]),
        ("AlphaFold DB model coordinates (PDB)", "burial feature",
         AFDB_FILE.format(accession="{accession}", kind=f"model_v{AFDB_MODEL_VERSION}.pdb"),
         f"AlphaFold DB model v{AFDB_MODEL_VERSION}",
         [f"contact number: Cβ (Cα for Gly) atoms within {BURIAL_RADIUS_A:g} Å of the site's Cβ"]),
        ("AlphaMissense tables (AlphaFold DB)", "variant-effect prior (human proteins only)",
         AFDB_FILE.format(accession="{accession}", kind="aa-substitutions.csv"),
         "AlphaMissense (Cheng et al. 2023, Science 381, eadg7492) as served by AlphaFold DB; no version number exposed",
         ["pathogenicity of the exact substitution", "class codes LBen/Amb/LPath -> likely_benign/ambiguous/likely_pathogenic",
          "mean of the 19 substitutions at the site"]),
        ("UniProt REST (FASTA)", "fallback reference sequence when AlphaFold DB has no model", UNIPROT_FASTA,
         "UniProtKB release current at the access time", ["first FASTA record, used only to validate numbering"]),
    ]
    for name, role, endpoint, version, tr in services:
        entry = {"name": name, "role": role, "type": "API", "endpoint": endpoint, "version": version,
                 **st[name].as_dict(), "transformations": tr}
        if name == "AlphaFold DB prediction API":
            entry["models"] = models
            entry["accessions_without_model"] = no_model
        datasets.append(entry)
    out = {
        "title": "Data provenance for the AF-ΔΔG project (PHYS 7440, Assignment 9)",
        "author": AUTHOR,
        "pipeline": {"module": "data_pipeline_ly.py", "name": PIPELINE_NAME, "version": PIPELINE_VERSION,
                     "module_sha256": file_sha256(Path(__file__)), "command": "python " + " ".join(argv),
                     "run_started_utc": started, "run_finished_utc": _utcnow(), "platform": platform.platform(),
                     "packages": _package_versions(),
                     "retry_policy": {"library": "tenacity", "max_attempts": fetcher.max_attempts,
                                      "retries": fetcher.max_attempts - 1,
                                      "wait": f"exponential {fetcher.wait.base:g}·2^(n-1) s (HTTP 429: "
                                              f"{fetcher.wait.rate_limit_base:g}·2^(n-1) s), cap {fetcher.wait.cap:g} s, "
                                              "Retry-After honoured",
                                      "retried_on": "timeouts, connection errors, HTTP 429, HTTP 5xx",
                                      "politeness": f"<= 1 request per {fetcher.min_interval:g} s per host; interval "
                                                    f"doubled after HTTP 429 (max {fetcher.max_interval:g} s)",
                                      "rate_limited_responses_by_host": dict(fetcher.rate_limited),
                                      "final_request_interval_s_by_host": {h: round(v, 3) for h, v in
                                                                           fetcher.host_interval.items()}},
                     "github_commits_api": st["GitHub commits API"].as_dict()},
        "datasets": datasets,
        "outputs": {
            "variants": {"path": _rel(paths.parquet, paths.root), "csv_copy": _rel(paths.csv, paths.root),
                         "rows": int(len(df)),
                         "rows_by_dataset": {k: int(v) for k, v in df["dataset"].value_counts().sort_index().items()},
                         "content_sha256": content,
                         "content_hash_definition": "SHA-256 of df[schema columns].to_csv(index=False, float_format='%.10g', "
                                                    "lineterminator='\\n', na_rep='NaN'); rows: S669 then S2648, each in source order",
                         "parquet_file_sha256": file_sha256(paths.parquet)},
            "missing_fields": {"path": _rel(paths.missing, paths.root), "events": len(mlog.events),
                               "substituted_values_by_field": mlog.summary()},
            "cache": {"path": _rel(paths.cache, paths.root), "responses": fetcher.cache.count()},
            "log": _rel(paths.logfile, paths.root),
        },
        "coverage": {
            ds: {"variants": int(len(g)),
                 "with_uniprot_position": int((g["position_uniprot"] != SENTINEL_INT).sum()),
                 "wild_type_verified": int((g["wt_check"] == "match").sum()),
                 "with_alphafold_plddt": int((g["plddt"] != SENTINEL_FLOAT).sum()),
                 "with_burial": int((g["burial_cn10"] != SENTINEL_INT).sum()),
                 "with_alphamissense": int((g["am_pathogenicity"] != SENTINEL_FLOAT).sum())}
            for ds, g in df.groupby("dataset", sort=True)},
        "schema": {c: {"dtype": t, "unit": u, "description": d,
                       "sentinel": (SENTINELS["ddg_kcal_mol"] if c == "ddg_kcal_mol"
                                    else None if c in STATUS_COLUMNS else sentinel_for(c))}
                   for c, (t, u, d) in SCHEMA.items()},
        "sentinels": SENTINELS,
    }
    prev = ((previous or {}).get("outputs") or {}).get("variants", {}).get("content_sha256")
    out["reproducibility"] = {"previous_content_sha256": prev,
                              "identical_to_previous": None if prev is None else prev == content,
                              "differences_from_previous_table": diff,
                              "how_to_check": "python data_pipeline_ly.py --fresh (re-downloads everything and compares "
                                              "with the content hash recorded before), then python data_pipeline_ly.py --verify"}
    return out


# ======================================================================================
# Orchestration
# ======================================================================================


def run(root: Path = HERE, *, fresh: bool = False, offline: bool = False, transport: httpx.BaseTransport | None = None,
        backoff_base: float = 1.0, max_attempts: int = 5, timeout: float = 30.0, min_interval: float = 0.25,
        contact: str | None = None, check_upstream: bool = True, argv: list[str] | None = None) -> dict[str, Any]:
    """Run the whole pipeline and return the provenance record.

    Steps: dataset files -> normalise -> SIFTS -> AlphaFold DB records -> UniProt
    numbering -> structure files -> features -> outputs -> provenance.
    """
    paths = Paths(Path(root))
    started = _utcnow()
    try:
        previous = json.loads(paths.provenance.read_text(encoding="utf-8")) if paths.provenance.exists() else None
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable %s (%s)", paths.provenance.name, exc)
        previous = None
    cache = ResponseCache(paths.cache)
    if fresh:
        log.info("--fresh: clearing %d cached responses", cache.count())
        cache.clear()
    fetcher = HttpFetcher(cache, offline=offline, transport=transport, timeout=timeout, max_attempts=max_attempts,
                          backoff_base=backoff_base, rate_limit_base=5.0 * backoff_base, min_interval=min_interval,
                          contact=contact)
    mlog = MissingFieldLog()
    try:
        # 1. dataset files
        frames, file_meta, upstream = {}, {}, {}
        for src in FILE_SOURCES:
            frames[src.name], file_meta[src.name] = fetch_file_source(src, fetcher, paths.raw)
            upstream[src.name] = (upstream_latest_commit(src, fetcher) if check_upstream and not offline
                                  else {"sha": SENTINEL_STR, "date": SENTINEL_STR, "note": "not checked in this run"})
            log.info("%s: %d rows (%s, SHA-256 OK)", src.name, file_meta[src.name]["rows"],
                     "from cache" if file_meta[src.name]["served_from_cache"] else "downloaded")
        # 2. normalise rows
        test_rows = normalise_s669(frames["S669"], S669, mlog)
        train_rows = normalise_s2648(frames["S2648"], S2648, mlog)
        sign = {"S669": check_sign_convention(test_rows, "S669"), "S2648": check_sign_convention(train_rows, "S2648")}
        # 3. SIFTS for every S2648 PDB entry
        chains = sorted({(r["_pdb"], r["_chain"]) for r in train_rows if r["_pdb"]})
        sifts_payloads, sifts_used = fetch_sifts(sorted({p for p, _ in chains}), fetcher)
        segments = {f"{p}_{c}": parse_sifts(sifts_payloads.get(p), p, c) for p, c in chains}
        sifts_urls = {f"{p}_{c}": sifts_used[p] for p, c in chains}
        n_mapped = sum(bool(s) for s in segments.values())
        log.info("SIFTS: %d of %d PDB chains mapped to UniProt", n_mapped, len(chains))
        if chains and n_mapped == 0 and any(v is not None for v in sifts_payloads.values()):
            log.warning("SIFTS answered but no chain could be parsed: the PDBe response format may have changed")
        # 4. AlphaFold DB prediction records (they also provide the reference sequences)
        accessions = sorted({r["uniprot_accession"] for r in test_rows if r["uniprot_accession"] != SENTINEL_STR}
                            | {s.accession for segs in segments.values() for s in segs})
        proteins: dict[str, ProteinData] = {}
        for k, acc in enumerate(accessions, 1):
            proteins[acc] = fetch_prediction(acc, fetcher)
            if not proteins[acc].sequence:
                fetch_uniprot_sequence(proteins[acc], fetcher)
            if k % 25 == 0 or k == len(accessions):
                log.info("AlphaFold DB records: %d/%d accessions", k, len(accessions))
        # 5. numbering: S2648 PDB -> UniProt; S669 wild types re-checked against the current UniProt sequence
        map_pdb_to_uniprot(train_rows, segments, {a: p.sequence for a, p in proteins.items()}, sifts_urls, mlog)
        for r in test_rows:
            p = proteins.get(r["uniprot_accession"])
            if p and p.sequence and r["position_uniprot"] != SENTINEL_INT:
                pos = r["position_uniprot"]
                if not (0 < pos <= len(p.sequence) and p.sequence[pos - 1] == r["wt"]):
                    r["wt_check"] = "mismatch"
                    mlog.record(r["variant_id"], ["wt_check"], p.prediction_url,
                                f"wild type {r['wt']} not at position {pos} of the current UniProt sequence")
        # 6. structure files for accessions with at least one usable variant
        needed = sorted({r["uniprot_accession"] for r in test_rows + train_rows
                         if r["uniprot_accession"] in proteins and proteins[r["uniprot_accession"]].record is not None
                         and r["position_uniprot"] != SENTINEL_INT and r["wt_check"] != "mismatch"})
        for k, acc in enumerate(needed, 1):
            fetch_structure_files(proteins[acc], fetcher)
            if k % 25 == 0 or k == len(needed):
                log.info("AlphaFold DB files: %d/%d proteins", k, len(needed))
        # 7. features
        for r in test_rows + train_rows:
            add_structure_features(r, proteins.get(r["uniprot_accession"]), mlog)
        df = to_frame([_finish_missing(r) for r in test_rows + train_rows])
        # 8. outputs and provenance (compare with the previous table first, for the reproducibility record)
        try:
            old_df = pd.read_parquet(paths.parquet) if paths.parquet.exists() else None
        except Exception as exc:                                  # unreadable previous file: just skip the diff
            log.warning("could not read the previous %s (%s)", paths.parquet.name, exc)
            old_df = None
        diff = diff_tables(old_df, df)
        paths.processed.mkdir(parents=True, exist_ok=True)
        df.to_parquet(paths.parquet, index=False, engine="pyarrow")
        df.to_csv(paths.csv, index=False, float_format="%.10g", lineterminator="\n", na_rep="NaN")
        mlog.as_frame().to_csv(paths.missing, index=False, lineterminator="\n")
        prov = build_provenance(paths=paths, df=df, file_meta=file_meta, upstream=upstream, fetcher=fetcher,
                                proteins=proteins, mlog=mlog, sign_fractions=sign, started=started,
                                argv=argv if argv is not None else ["data_pipeline_ly.py"], previous=previous, diff=diff)
        paths.provenance.write_text(json.dumps(prov, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    finally:
        fetcher.close()
        cache.close()
    failed = sum(s.failed for name, s in fetcher.stats.items() if name != "GitHub commits API")
    log.info("wrote %s (%d rows), content SHA-256 %s", _rel(paths.parquet, paths.root), len(df),
             prov["outputs"]["variants"]["content_sha256"])
    log.info("missing-field events: %d (details in %s and %s)", len(mlog.events), _rel(paths.missing, paths.root),
             _rel(paths.logfile, paths.root))
    for ds, cov in prov["coverage"].items():
        log.info("%s coverage: %s", ds, cov)
    log.info("network: %d live requests, %d served from cache, %d failed after retries", fetcher.live_calls,
             sum(s.from_cache for s in fetcher.stats.values()), failed)
    rep = prov["reproducibility"]
    if rep["identical_to_previous"] is True:
        log.info("REPRODUCIBLE: content hash identical to the previous provenance record")
    elif rep["identical_to_previous"] is False:
        log.warning("content hash DIFFERS from the previous provenance record (%s); differences: %s",
                    rep["previous_content_sha256"], rep["differences_from_previous_table"])
    if failed:
        log.warning("%d requests failed after all retries; their fields are sentinels. Re-run to fill them in.", failed)
    return prov


def verify(root: Path = HERE) -> bool:
    """Check the outputs against provenance_ly.json without any network access; print a report."""
    paths = Paths(Path(root))
    prov = json.loads(paths.provenance.read_text(encoding="utf-8"))
    df = pd.read_parquet(paths.parquet)
    got, want = content_sha256(df), prov["outputs"]["variants"]["content_sha256"]
    ok = got == want
    print(f"variants content SHA-256: {got} -> {'OK' if ok else 'MISMATCH (provenance has ' + want + ')'}")
    for entry in prov["datasets"]:
        if entry.get("type") == "file":
            src = next(s for s in FILE_SOURCES if s.name == entry["name"])
            path = paths.raw / src.local_name
            digest = file_sha256(path) if path.exists() else "absent"
            print(f"{src.name} raw file SHA-256: {'OK' if digest == src.sha256 else 'MISMATCH'}")
            ok &= digest == src.sha256
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    print("schema columns:", "OK" if not missing_cols else f"missing {missing_cols}")
    return ok and not missing_cols


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="AF-ΔΔG data ingestion pipeline (Assignment 9).")
    ap.add_argument("--fresh", action="store_true", help="clear the HTTP cache first (simulates a fresh clone)")
    ap.add_argument("--offline", action="store_true", help="never use the network; cache misses become sentinels")
    ap.add_argument("--verify", action="store_true", help="check outputs against provenance_ly.json and exit")
    ap.add_argument("--root", type=Path, default=HERE, help="project folder (default: this file's folder)")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds (default 30)")
    ap.add_argument("--max-attempts", type=int, default=5, help="attempts per request, >= 4 (default 5 = 4 retries)")
    ap.add_argument("--contact", default=None, help="e-mail added to the User-Agent (optional, polite)")
    ap.add_argument("--no-upstream-check", action="store_true", help="skip the GitHub latest-commit lookup")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print every missing-field event")
    args = ap.parse_args(argv)
    if args.verify:
        if not Paths(args.root).provenance.exists() or not Paths(args.root).parquet.exists():
            print("nothing to verify yet: run `python data_pipeline_ly.py` first")
            return 1
        return 0 if verify(args.root) else 1
    setup_logging(Paths(args.root).logfile, args.verbose)
    try:
        run(args.root, fresh=args.fresh, offline=args.offline, timeout=args.timeout, max_attempts=args.max_attempts,
            contact=args.contact, check_upstream=not args.no_upstream_check, argv=["data_pipeline_ly.py", *argv])
    except (UpstreamError, RuntimeError) as exc:
        # only the dataset files are essential: without them (or with changed checksums) nothing can be built
        log.error("pipeline stopped: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
