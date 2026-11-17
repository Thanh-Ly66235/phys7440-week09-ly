"""Offline tests for data_pipeline_ly.py (no network needed).

Run from the project folder:  python -m pytest -q

The HTTP layer is exercised through ``httpx.MockTransport``. The AlphaFold DB
payloads for P68871 (HBB) and P04637 (TP53) are the real responses recorded for
my Assignment 7 MCP server (tests/fixtures/afdb), so the consistency test checks
that this pipeline extracts exactly the pLDDT and AlphaMissense values that the
Assignment 7 server reported.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data_pipeline_ly as dp  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "afdb"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def make_fetcher(tmp_path, handler, **kw):
    """HttpFetcher on a fresh cache, with zero backoff so the tests run instantly."""
    cache = dp.ResponseCache(tmp_path / "cache.sqlite")
    kw.setdefault("backoff_base", 0.0)
    kw.setdefault("rate_limit_base", 0.0)
    kw.setdefault("min_interval", 0.0)
    kw.setdefault("max_interval", 0.0)
    return dp.HttpFetcher(cache, transport=httpx.MockTransport(handler), **kw)


def fixture_json(name):
    """Parsed JSON fixture."""
    return json.loads((FIX / name).read_text())


def synthetic_pdb(sequence: str, spacing: float = 3.8) -> str:
    """A PDB text with CA and CB atoms on a compact 3D lattice (good enough for contact counts)."""
    three = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
             "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
             "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}
    lines, serial = [], 1
    side = max(2, round(len(sequence) ** (1 / 3)) + 1)
    for i, aa in enumerate(sequence, 1):
        x, y, z = (i - 1) % side, ((i - 1) // side) % side, (i - 1) // (side * side)
        ca = np.array([x, y, z], float) * spacing
        for name, xyz in (("CA", ca), ("CB", ca + 1.0)):
            if name == "CB" and aa == "G":
                continue
            lines.append(f"ATOM  {serial:5d}  {name:<3s} {three[aa]} A{i:4d}    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 90.00           C")
            serial += 1
    return "\n".join(lines + ["END"]) + "\n"


# --------------------------------------------------------------------------------------
# failure mode (a): timeouts, rate limits, server errors -> tenacity retries
# --------------------------------------------------------------------------------------


def test_timeout_is_retried_then_succeeds(tmp_path, caplog):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        return httpx.Response(200, json={"ok": True})

    f = make_fetcher(tmp_path, handler)
    with caplog.at_level(logging.WARNING, logger="af_ddg"):
        r = f.fetch("https://example.org/a", "test")
    assert r.status == 200 and r.attempts == 3 and calls["n"] == 3
    assert f.stats["test"].retries == 2
    assert sum("retry" in m for m in caplog.messages) == 2


def test_rate_limit_429_is_retried(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, text="fine")

    f = make_fetcher(tmp_path, handler)
    r = f.fetch("https://example.org/b", "test", accept="text/plain")
    assert r.status == 200 and r.attempts == 2


def test_rate_limit_slows_down_that_host_only(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429) if calls["n"] == 1 else httpx.Response(200, json={})

    f = make_fetcher(tmp_path, handler, min_interval=0.0, max_interval=2.0)
    f.fetch("https://slow.example.org/a", "test")
    assert f.host_interval["slow.example.org"] == 0.5 and f.rate_limited["slow.example.org"] == 1
    assert f.host_interval["other.example.org"] == 0.0


def test_gives_up_after_max_attempts_without_caching(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    f = make_fetcher(tmp_path, handler)
    with pytest.raises(dp.UpstreamError) as exc:
        f.fetch("https://example.org/c", "test")
    assert calls["n"] == 5                       # 1 try + 4 retries
    assert "failed after 5 attempts" in exc.value.reason
    assert f.cache.get("https://example.org/c") is None
    assert f.stats["test"].failed == 1


def test_circuit_opens_after_repeated_host_failures(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("host down", request=request)

    f = make_fetcher(tmp_path, handler, circuit_threshold=2)
    for k in range(4):
        with pytest.raises(dp.UpstreamError):
            f.fetch(f"https://down.example.org/{k}", "test")
    assert calls["n"] == 2 * 5                     # two URLs tried 5 times each, the other two skipped
    assert f.stats["test"].failed == 4


def test_at_least_three_retries_is_enforced(tmp_path):
    with pytest.raises(ValueError):
        make_fetcher(tmp_path, lambda r: httpx.Response(200), max_attempts=3)


def test_backoff_is_exponential_and_honours_retry_after():
    wait = dp.BackoffWait(base=1.0, cap=60.0)

    def state(n, exc=None):
        outcome = SimpleNamespace(exception=lambda: exc)
        return SimpleNamespace(attempt_number=n, outcome=outcome)

    assert [wait(state(n)) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 8.0]
    assert [wait(state(n, dp.RetryableHTTPStatus("u", 503, None))) for n in (1, 2)] == [1.0, 2.0]
    assert [wait(state(n, dp.RetryableHTTPStatus("u", 429, None))) for n in (1, 2, 3, 4)] == [5.0, 10.0, 20.0, 40.0]
    assert wait(state(10)) == 60.0                                               # capped
    assert wait(state(1, dp.RetryableHTTPStatus("u", 429, 30.0))) == 30.0      # Retry-After wins


def test_404_is_cached_and_second_call_uses_cache(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404, json={})

    f = make_fetcher(tmp_path, handler)
    assert f.fetch("https://example.org/d", "test").status == 404
    r = f.fetch("https://example.org/d", "test")
    assert r.from_cache and calls["n"] == 1


def test_corrupt_cached_json_is_evicted(tmp_path):
    f = make_fetcher(tmp_path, lambda r: httpx.Response(200, text="<html>Service temporarily unavailable</html>"))
    with pytest.raises(dp.UpstreamError):
        f.fetch_json("https://example.org/j", "test")
    assert f.cache.get("https://example.org/j") is None          # a later run will fetch it again


def test_contact_numbers_chunked_matches_direct():
    rng = np.random.default_rng(0)
    centres = {i: rng.uniform(0, 40, 3) for i in range(1, 600)}
    xyz = np.stack([centres[i] for i in sorted(centres)])
    d2 = ((xyz[:, None] - xyz[None]) ** 2).sum(-1)
    direct = (d2 <= 100.0).sum(1) - 1
    assert list(dp.contact_numbers(centres).values()) == direct.tolist()


def test_cli_verify_without_outputs_fails_cleanly(tmp_path, capsys):
    assert dp.main(["--verify", "--root", str(tmp_path)]) == 1
    assert "run `python data_pipeline_ly.py` first" in capsys.readouterr().out


def test_offline_cache_miss_raises(tmp_path):
    f = make_fetcher(tmp_path, lambda r: httpx.Response(200), offline=True)
    with pytest.raises(dp.UpstreamError):
        f.fetch("https://example.org/e", "test")


# --------------------------------------------------------------------------------------
# failure mode (b): missing / null fields -> logged with URL, sentinel substituted
# --------------------------------------------------------------------------------------


def _hbb_protein(plddt=None, am=True):
    """ProteinData for HBB built from the recorded Assignment 7 fixtures."""
    rec = dp.choose_afdb_record(fixture_json("prediction_P68871.json"), "P68871")
    p = dp.ProteinData("P68871", dp.AFDB_PREDICTION.format(accession="P68871"), record=rec,
                       sequence=rec["uniprotSequence"])
    p.plddt_url = dp.afdb_file_url(rec, "P68871", "confidence")
    p.plddt = dp.parse_confidence(fixture_json("confidence_P68871.json")) if plddt is None else plddt
    p.model_url = dp.afdb_file_url(rec, "P68871", "model")
    p.burial = dp.contact_numbers(dp.parse_pdb_side_chain_centres(synthetic_pdb(rec["sequence"])))
    p.am_url = dp.afdb_file_url(rec, "P68871", "aa-substitutions")
    p.am = dp.parse_alphamissense((FIX / "am_P68871.csv").read_text()) if am else None
    if not am:
        p.am_url, p.am_error = None, "no AlphaMissense table in AlphaFold DB (AlphaMissense covers human proteins only)"
    return p


def _row(accession="P68871", wt="E", pos=7, mt="V"):
    """A normalised row ready for add_structure_features."""
    row = dp._empty_row()
    row.update(variant_id="T:0000", uniprot_accession=accession, wt=wt, mt=mt, position_uniprot=pos,
               wt_check="match", _source_url="https://example.org/source.csv")
    return row


def test_null_plddt_becomes_sentinel_and_is_logged_with_url():
    p = _hbb_protein()
    p.plddt = {k: v for k, v in p.plddt.items() if k != 7}         # residue 7 is null in this response
    row, mlog = _row(), dp.MissingFieldLog()
    dp.add_structure_features(row, p, mlog)
    assert row["plddt"] == dp.SENTINEL_FLOAT and row["plddt_window"] == dp.SENTINEL_FLOAT
    ev = [e for e in mlog.events if e["field"] == "plddt"]
    assert ev and ev[0]["source_url"] == p.plddt_url and "null" in ev[0]["reason"]
    assert row["am_pathogenicity"] == 0.2229                         # other fields still filled


def test_parse_confidence_skips_nulls():
    out = dp.parse_confidence({"residueNumber": [1, 2, 3], "confidenceScore": [90.0, None, "x"]})
    assert out == {1: 90.0}


def test_non_human_protein_gets_alphamissense_sentinels():
    p = _hbb_protein(am=False)
    row, mlog = _row(), dp.MissingFieldLog()
    dp.add_structure_features(row, p, mlog)
    assert row["am_pathogenicity"] == dp.SENTINEL_FLOAT and row["am_class"] == dp.SENTINEL_STR
    assert row["plddt"] == 96.56
    assert {e["field"] for e in mlog.events} == set(dp.AM_FIELDS)
    assert all(e["source_url"] == p.prediction_url for e in mlog.events)


def test_no_alphafold_model_sets_every_af_field_to_sentinel():
    p = dp.ProteinData("P99999", dp.AFDB_PREDICTION.format(accession="P99999"),
                       record_error="AlphaFold DB has no model for P99999 (HTTP 404)")
    row, mlog = _row("P99999"), dp.MissingFieldLog()
    dp.add_structure_features(row, p, mlog)
    dp._finish_missing(row)
    assert all(dp._is_sentinel(f, row[f]) for f in dp.AF_FIELDS)
    assert set(row["missing_fields"].split(";")) >= set(dp.AF_FIELDS)
    assert {e["source_url"] for e in mlog.events} == {p.prediction_url}


def test_sentinels_to_nan_only_touches_numeric_sentinels():
    df = pd.DataFrame({"plddt": [-1.0, 80.0], "burial_cn10": [-1, 12], "am_class": ["MISSING", "ambiguous"]})
    out = dp.sentinels_to_nan(df)
    assert math.isnan(out.loc[0, "plddt"]) and math.isnan(out.loc[0, "burial_cn10"]) and out.loc[1, "burial_cn10"] == 12
    assert out.loc[0, "am_class"] == "MISSING"


# --------------------------------------------------------------------------------------
# features, parsing and identifier normalisation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("acc, variant, plddt, am, klass", [
    ("P68871", "E7V", 96.56, 0.2229, "likely_benign"),        # sickle-cell variant (Assignment 7 example)
    ("P04637", "R175H", 96.62, 0.9857, "likely_pathogenic"),  # TP53 hotspot (Assignment 7 example)
])
def test_features_match_assignment7_server_output(acc, variant, plddt, am, klass):
    rec = dp.choose_afdb_record(fixture_json(f"prediction_{acc}.json"), acc)
    assert rec["entryId"] == f"AF-{acc}-F1"                    # canonical isoform chosen, as in Assignment 7
    p = dp.ProteinData(acc, dp.AFDB_PREDICTION.format(accession=acc), record=rec, sequence=rec["uniprotSequence"])
    p.plddt = dp.parse_confidence(fixture_json(f"confidence_{acc}.json"))
    p.burial = dp.contact_numbers(dp.parse_pdb_side_chain_centres(synthetic_pdb(rec["sequence"])))
    p.am = dp.parse_alphamissense((FIX / f"am_{acc}.csv").read_text())
    wt, pos, _, mt = dp.parse_mutation(variant)
    row, mlog = _row(acc, wt, pos, mt), dp.MissingFieldLog()
    dp.add_structure_features(row, p, mlog)
    assert (row["plddt"], row["am_pathogenicity"], row["am_class"]) == (plddt, am, klass)
    assert row["af_model_version"] == 6 and not mlog.events


def test_parse_mutation():
    assert dp.parse_mutation("S11A") == ("S", 11, "", "A")
    assert dp.parse_mutation("L27CN") == ("L", 27, "C", "N")
    assert dp.parse_mutation("X5A") is None and dp.parse_mutation("junk") is None


def test_contact_numbers_simple_geometry():
    centres = {1: np.zeros(3), 2: np.array([5.0, 0, 0]), 3: np.array([9.0, 0, 0]), 4: np.array([30.0, 0, 0])}
    assert dp.contact_numbers(centres, 10.0) == {1: 2, 2: 2, 3: 2, 4: 0}


def test_glycine_uses_calpha():
    pdb = synthetic_pdb("AGA")
    centres = dp.parse_pdb_side_chain_centres(pdb)
    assert sorted(centres) == [1, 2, 3]


def _sifts(pdb, acc, chain, author_start, author_end, unp_start, unp_end):
    """Minimal PDBe SIFTS payload in the documented format."""
    return {pdb.lower(): {"UniProt": {acc: {"identifier": "X", "name": "X", "mappings": [{
        "entity_id": 1, "chain_id": chain, "struct_asym_id": chain,
        "start": {"author_residue_number": author_start, "author_insertion_code": "", "residue_number": 1},
        "end": {"author_residue_number": author_end, "author_insertion_code": "", "residue_number": 99},
        "unp_start": unp_start, "unp_end": unp_end}]}}}}


def test_sifts_offset_mapping_vote_and_insertion_code():
    seq = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKL"
    segs = dp.parse_sifts(_sifts("1ABC", "P68871", "A", 1, 90, 2, 91), "1ABC", "A")
    assert segs[0].offset == 1
    rows = []
    for i, mut in enumerate(["V1A", "E6Q", "T12A", "L27CN", "W36F"]):
        r = dp._empty_row()
        wt, n, ic, mt = dp.parse_mutation(mut)
        r.update(variant_id=f"S:{i}", protein="1ABC_A", wt=wt, mt=mt, _resnum=n, _icode=ic)
        rows.append(r)
    mlog = dp.MissingFieldLog()
    dp.map_pdb_to_uniprot(rows, {"1ABC_A": segs}, {"P68871": seq}, {"1ABC_A": "https://sifts/1abc"}, mlog)
    got = {r["variant_id"]: (r["position_uniprot"], r["wt_check"]) for r in rows}
    assert got["S:0"] == (2, "match") and got["S:1"] == (7, "match") and got["S:2"] == (13, "match")
    assert got["S:3"][0] == dp.SENTINEL_INT                         # insertion code -> unmapped, logged
    assert any("insertion code" in e["reason"] for e in mlog.events)


def test_sifts_null_author_end_is_inferred():
    payload = _sifts("1FVK", "P0AEG4", "A", 1, None, 20, 208)          # as served by PDBe for 1FVK
    seg = dp.parse_sifts(payload, "1FVK", "A")[0]
    assert (seg.offset, seg.author_end) == (19, 189) and seg.covers(30)


def test_offset_vote_fixes_a_wrong_sifts_offset():
    seq = "MKTAYIAKQRQISFVKSHFSRQ"
    # PDB numbering is UniProt - 3, but the (deliberately wrong) segment says offset 0
    segs = dp.parse_sifts(_sifts("2XYZ", "Q00001", "B", 1, 30, 1, 30), "2XYZ", "B")
    rows = []
    for i, uni in enumerate([5, 8, 12, 16]):
        r = dp._empty_row()
        r.update(variant_id=f"S:{i}", protein="2XYZ_B", wt=seq[uni - 1], mt="A" if seq[uni - 1] != "A" else "G",
                 _resnum=uni - 3, _icode="")
        rows.append(r)
    dp.map_pdb_to_uniprot(rows, {"2XYZ_B": segs}, {"Q00001": seq}, {}, dp.MissingFieldLog())
    assert [r["position_uniprot"] for r in rows] == [5, 8, 12, 16]
    assert {r["numbering_map"] for r in rows} == {"SIFTS accession + chain-wide offset vote"}


def test_diff_tables_locates_changes():
    rows = [dp._finish_missing(dict(dp._empty_row(), variant_id=f"S669:{i:04d}", dataset="S669", split="test",
                                    source_row=i, plddt=80.0 + i)) for i in range(3)]
    old = dp.to_frame(rows)
    new = old.copy()
    new.loc[1, "plddt"] = 55.0
    d = dp.diff_tables(old, new.iloc[:2])
    assert d == {"rows_only_in_previous": 1, "rows_only_in_new": 0, "changed_values_by_column": {"plddt": 1}}
    assert dp.diff_tables(None, new) is None


def test_s669_id_cleaning_and_pdb_chain_sentinel():
    df = pd.DataFrame({"ID": ["P04517WTE2143G", "1fh5H"], "MTS": ["E2143G", "S30A"], "DDG": ["-1.0", ""],
                       "Sequence": ["", "AAAA"]})
    mlog = dp.MissingFieldLog()
    rows = dp.normalise_s669(df, dp.S669, mlog)
    assert rows[0]["uniprot_accession"] == "P04517" and rows[0]["protein"] == "P04517"
    assert rows[1]["uniprot_accession"] == dp.SENTINEL_STR
    assert math.isnan(rows[1]["ddg_kcal_mol"])                      # blank ΔΔG -> NaN sentinel, logged
    fields = {(e["variant_id"], e["field"]) for e in mlog.events}
    assert ("S669:0001", "uniprot_accession") in fields and ("S669:0001", "ddg_kcal_mol") in fields


def test_s2648_temperature_converted_to_kelvin():
    df = pd.DataFrame({"CLID": ["c"], "PDB": ["1abc"], "CHAIN": ["A"], "MUT": ["V1A"], "DDG": ["-1.2"],
                       "pH": ["7.0"], "T": ["25.0"]})
    row = dp.normalise_s2648(df, dp.S2648, dp.MissingFieldLog())[0]
    assert row["temperature_k"] == 298.15 and row["ph"] == 7.0 and row["protein"] == "1ABC_A"


# --------------------------------------------------------------------------------------
# end to end, against fake services (tiny datasets, real recorded AlphaFold payloads)
# --------------------------------------------------------------------------------------


def _tiny_sources():
    """Two small dataset files plus FileSource objects pointing at fake URLs."""
    hbb = fixture_json("prediction_P68871.json")[0]["uniprotSequence"]
    p53 = fixture_json("prediction_P04637.json")[0]["uniprotSequence"]
    s669 = ("ID,MTS,DDG,Sequence\n"
            f"P68871,E7V,-0.5,{hbb}\n"
            f"P04637,R175H,-2.0,{p53}\n"
            "P99999,A2G,-1.1,MAAAA\n"
            "1fh5H,S30A,-0.7,AAAA\n").encode()
    s2648 = ("CLID,PDB,CHAIN,MUT,DDG,pH,T\n"
             "c1,1ABC,A,E6V,-0.8,7.0,25.0\n"        # PDB numbering = UniProt - 1
             "c1,1ABC,A,V1A,-1.5,7.0,25.0\n"
             "c1,1ABC,A,T12S,-0.9,7.0,25.0\n"
             "c2,9ZZZ,A,L10A,-2.5,6.5,37.0\n").encode()   # SIFTS has no entry
    def src(base, name, role, body, numbering, tunit):
        return dp.FileSource(name=name, role=role, repo="fake/repo", commit="c0ffee", file_commit="c0ffee",
                             file_commit_date="2026-01-01", path=f"{name.lower()}.csv",
                             sha256=hashlib.sha256(body).hexdigest(), local_name=f"{name.lower()}.csv",
                             citation="test", doi="10.0/test", numbering=numbering, ddg_unit="kcal/mol",
                             ddg_sign=1, temperature_unit=tunit)
    return (src("S669", "S669", "test", s669, "UniProt", None), s669,
            src("S2648", "S2648", "train", s2648, "PDB", "degC"), s2648)


def _fake_services(s669_src, s669_body, s2648_src, s2648_body, flaky_counter):
    """MockTransport handler that imitates GitHub, PDBe, AlphaFold DB and UniProt."""
    preds = {"P68871": (FIX / "prediction_P68871.json").read_bytes(), "P04637": (FIX / "prediction_P04637.json").read_bytes()}
    seqs = {a: json.loads(b)[0]["sequence"] for a, b in preds.items()}

    def handler(request):
        url = str(request.url)
        if url == s669_src.url:
            return httpx.Response(200, content=s669_body)
        if url == s2648_src.url:
            return httpx.Response(200, content=s2648_body)
        if "api.github.com" in url:
            return httpx.Response(200, json=[{"sha": "c0ffee", "commit": {"committer": {"date": "2026-01-01T00:00:00Z"}}}])
        if "/pdbe/api/mappings/uniprot/1abc" in url:
            return httpx.Response(404, json={})                          # primary API misses it ...
        if "/pdbe/graph-api/mappings/uniprot/1abc" in url:
            return httpx.Response(200, json=_sifts("1ABC", "P68871", "A", 1, 146, 2, 147))   # ... fallback has it
        if "/mappings/uniprot/" in url:
            return httpx.Response(404, json={})
        for acc, body in preds.items():
            if url.endswith(f"/api/prediction/{acc}"):
                return httpx.Response(200, content=body)
            if url.endswith(f"AF-{acc}-F1-confidence_v6.json"):
                return httpx.Response(200, content=(FIX / f"confidence_{acc}.json").read_bytes())
            if url.endswith(f"AF-{acc}-F1-aa-substitutions.csv"):
                return httpx.Response(200, content=(FIX / f"am_{acc}.csv").read_bytes())
            if url.endswith(f"AF-{acc}-F1-model_v6.pdb"):
                if acc == "P68871":
                    flaky_counter["n"] += 1
                    if flaky_counter["n"] <= 2:                          # two server errors, then success
                        return httpx.Response(503)
                return httpx.Response(200, text=synthetic_pdb(seqs[acc]))
        if "/api/prediction/" in url:
            return httpx.Response(404, json={})
        if "rest.uniprot.org" in url:
            return httpx.Response(200, text=">sp|P99999|TEST\nMAAAA\n")
        return httpx.Response(404, text="unknown")
    return handler


def test_end_to_end_reproducible_and_cached(tmp_path, monkeypatch):
    s669_src, s669_body, s2648_src, s2648_body = _tiny_sources()
    monkeypatch.setattr(dp, "S669", s669_src)
    monkeypatch.setattr(dp, "S2648", s2648_src)
    monkeypatch.setattr(dp, "FILE_SOURCES", (s669_src, s2648_src))
    counter = {"n": 0}
    handler = _fake_services(s669_src, s669_body, s2648_src, s2648_body, counter)
    prov1 = dp.run(tmp_path, transport=httpx.MockTransport(handler), backoff_base=0.0, min_interval=0.0)
    df = dp.load_variants(tmp_path)
    assert len(df) == 8 and list(df.columns) == dp.COLUMNS
    hbb = df[(df.dataset == "S669") & (df.mutation_uniprot == "E7V")].iloc[0]
    assert hbb.plddt == 96.56 and hbb.am_pathogenicity == 0.2229 and hbb.burial_cn10 >= 0
    # S2648 PDB numbering mapped to UniProt numbering (offset +1) and joined with the same features
    tr = df[df.dataset == "S2648"].set_index("source_mutation")
    assert tr.loc["E6V", "mutation_uniprot"] == "E7V" and tr.loc["E6V", "plddt"] == 96.56
    assert tr.loc["L10A", "uniprot_accession"] == "MISSING" and tr.loc["L10A", "plddt"] == -1.0
    assert tr.loc["E6V", "temperature_k"] == 298.15
    # failure modes were exercised without crashing
    p68871_pdb = next(d for d in prov1["datasets"] if d["name"] == "AlphaFold DB model coordinates (PDB)")
    assert p68871_pdb["retries"] == 2 and p68871_pdb["failed_after_retries"] == 0
    missing = pd.read_csv(tmp_path / "data" / "processed" / "missing_fields_ly.csv")
    assert (missing.source_url.str.contains("/pdbe/api/mappings/uniprot/9zzz")).any()
    assert (missing.source_url.str.contains("/api/prediction/P99999")).any()
    # provenance: every dataset has a source, an access time, a version/commit and transformations
    for d in prov1["datasets"]:
        assert (d.get("source_url") or d.get("endpoint")) and d["transformations"]
        assert d.get("commit_pinned_in_url") or d.get("version")
        if d["type"] == "file" or d["requests"]:
            assert d.get("access_timestamp_utc") or d.get("first_access_utc")
    assert prov1["reproducibility"]["identical_to_previous"] is None
    # second run: everything comes from the cache and the content is identical
    def no_network(request):
        raise AssertionError(f"unexpected network call: {request.url}")
    prov2 = dp.run(tmp_path, transport=httpx.MockTransport(no_network), check_upstream=False,
                   backoff_base=0.0, min_interval=0.0)
    assert prov2["outputs"]["variants"]["content_sha256"] == prov1["outputs"]["variants"]["content_sha256"]
    assert prov2["reproducibility"]["identical_to_previous"] is True
    assert prov2["reproducibility"]["differences_from_previous_table"] == {
        "rows_only_in_previous": 0, "rows_only_in_new": 0, "changed_values_by_column": {}}
    assert dp.verify(tmp_path)
    # fresh run (cache cleared, everything downloaded again) reproduces the same content
    counter["n"] = 0
    prov3 = dp.run(tmp_path, fresh=True, transport=httpx.MockTransport(handler), backoff_base=0.0, min_interval=0.0)
    assert prov3["outputs"]["variants"]["content_sha256"] == prov1["outputs"]["variants"]["content_sha256"]


def test_upstream_outage_does_not_crash_the_pipeline(tmp_path, monkeypatch):
    s669_src, s669_body, s2648_src, s2648_body = _tiny_sources()
    monkeypatch.setattr(dp, "S669", s669_src)
    monkeypatch.setattr(dp, "S2648", s2648_src)
    monkeypatch.setattr(dp, "FILE_SOURCES", (s669_src, s2648_src))
    good = _fake_services(s669_src, s669_body, s2648_src, s2648_body, {"n": 10})

    def outage(request):
        if "alphafold.ebi.ac.uk" in str(request.url):
            raise httpx.ConnectTimeout("AlphaFold DB unreachable", request=request)
        return good(request)

    prov = dp.run(tmp_path, transport=httpx.MockTransport(outage), backoff_base=0.0, min_interval=0.0)
    df = dp.load_variants(tmp_path)
    assert (df.plddt == -1.0).all()                                   # all AF features are sentinels
    afdb = next(d for d in prov["datasets"] if d["name"] == "AlphaFold DB prediction API")
    assert afdb["failed_after_retries"] == 3                          # P04637, P68871, P99999: each tried 5 times
