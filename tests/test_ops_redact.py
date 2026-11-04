"""Redaction corpus, idempotence, fingerprints, budget and handler output."""

from __future__ import annotations

import logging
import statistics
import time

import pytest

from lightrag.api.ops.redact import (
    redact_pipeline,
    EXC_MAX_CHARS,
    MSG_MAX_CHARS,
    RedactFilter,
    fingerprint,
    mask_path,
    normalise_message,
    redact,
    redact_traceback,
    text_ref,
)
from lightrag.kg.postgres_impl import describe_sql_params

pytestmark = pytest.mark.offline

JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJvcHNvQHVuaW1hcy5teSIsInJvbGUiOiJhZG1pbiJ9."
    "Zx3cQm9yqLk2R8sVt0uWw5yXz1aBcDeFgHiJkLmNoPq"
)
B64_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kg"
    "AAAABJRU5ErkJggg=="
)

# (text, substrings that must not survive)
CORPUS: list[tuple[str, list[str]]] = [
    (
        # postgres_impl.py execute() error line, in its historical shape
        "PostgreSQL database,\nsql:INSERT INTO LIGHTRAG_DOC_FULL ...,\n"
        "data:{'workspace': 'opso_unimas_my', 'id': 'doc-1', "
        "'content': 'Dear alice@unimas.my, IC 900101-13-5678, call 012-345 6789', "
        "'file_path': 'Senate_minutes_2026.pdf', 'password': 'hunter2'},\n"
        "error:duplicate key",
        [
            "alice@unimas.my",
            "900101-13-5678",
            "012-345 6789",
            "Senate_minutes_2026",
            "hunter2",
        ],
    ),
    (f"Authorization: Bearer {JWT}", [JWT, JWT[:40]]),
    (f"token refresh failed for {JWT}", [JWT]),
    ("calling provider with Bearer abcdef1234567890XYZ", ["abcdef1234567890XYZ"]),
    ("OPENAI_API_KEY=sk-proj-AbCdEf1234567890xyz loaded", ["AbCdEf1234567890xyz"]),
    ("key sk-ant-api03-ABCDEFGHIJKLMNOP leaked", ["ABCDEFGHIJKLMNOP"]),
    (
        '{"api_key": "abc123secret", "client_secret": "s3cr3t"}',
        ["abc123secret", "s3cr3t"],
    ),
    ("x-api-key: k-123456 and password=pa55word", ["k-123456", "pa55word"]),
    (
        "connect postgres://lightrag:Sup3rS3cret@postgres:5432/lightrag failed",
        ["Sup3rS3cret", "lightrag:Sup3r"],
    ),
    ("neo4j://neo4j:neo4jpass@neo4j:7687 unreachable", ["neo4jpass"]),
    (
        "Skipped LightRAG's own email from alert@unimas.my subject Re: budget",
        ["alert@unimas.my"],
    ),
    (f"vision input data:image/png;base64,{B64_IMAGE}", [B64_IMAGE[:30]]),
    ("github token ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcd", ["ghp_AbCd"]),
    ("sha 3f2a9b1c3f2a9b1c3f2a9b1c3f2a9b1c3f2a9b1c3f2a9b1c", ["3f2a9b1c3f2a9b1c3f2a"]),
    ("student IC 900101-13-5678 and 900101135678", ["900101-13-5678", "900101135678"]),
    (
        "phones +60 12-345 6789, 60123456789, 012-3456789, 019 876 5432",
        ["12-345 6789", "60123456789", "012-3456789", "019 876 5432"],
    ),
    ("card number 4111111111111111 rejected", ["4111111111111111"]),
    (
        "PDF extraction failed for email_3f2a9b1c__attachment_1_Budget_2026.xlsx",
        ["Budget_2026", "3f2a9b1c__"],
    ),
    ("Processing .eml file: Gaji_Disember_Ahmad.eml", ["Gaji_Disember_Ahmad"]),
    (
        # pipeline.py duplicate warning, historical shape
        "Duplicate document detected (filename): doc-0123abcd (Senate_minutes_2026.pdf)",
        ["Senate_minutes_2026"],
    ),
    (
        "Failed to extract document 3/12: /app/data/inputs/ws/Laporan_Kewangan.docx",
        ["Laporan_Kewangan", "/app/data/inputs/ws/"],
    ),
    (
        # Mixed-case path with digits: not mistaken for a base64 key, and
        # the file name is still masked.
        "Failed: /app/data/inputs/opso_unimas_my/Laporan_Kewangan_2024.pdf",
        ["Laporan_Kewangan_2024", "opso_unimas_my/"],
    ),
    # Quoted secrets keep nothing of their value, spaces and commas included.
    ("password=abc,def", ["abc", "def"]),
    ("PASSWORD: 'p@ss w0rd'", ["p@ss", "w0rd"]),
    ('{"client_secret": "a b, c", "x": 1}', ["a b", ", c"]),
    ("Authorization: 'Basic dXNlcjpwYXNz word'", ["dXNlcjpwYXNz", "word"]),
    # Quoted file names are masked whole, spaces included (OSError reprs,
    # Filename: "...").
    (
        "[Errno 2] No such file or directory: '/app/inputs/ws/Minit Mesyuarat "
        "Jabatan.pdf'",
        ["Minit", "Mesyuarat", "Jabatan", "/app/inputs/"],
    ),
    ('Filename: "Gaji Ahmad bin Ali.pdf"', ["Gaji", "Ahmad", "Ali"]),
    # Image and markup types email and uploads carry.
    (
        "Skipping inline image IC_AliBinAbu.jpg and scan_Siti.PNG, notes.md",
        ["AliBinAbu", "Siti", "notes"],
    ),
    # email_routes.py attachment / inline-image lines, historical shapes
    (
        "Failed to process attachment Slip_Gaji_Mac_2026.pdf: boom",
        ["Slip_Gaji_Mac"],
    ),
    (
        "Failed to process inline image Passport_Ahmad.jpg: decode error",
        ["Passport_Ahmad"],
    ),
    # pipeline.py extracting / completed lines, historical shapes
    ("Extracting stage 1/3: Senate_minutes_Sept_2026.pdf", ["Senate_minutes"]),
    (
        "Completed processing file 3/3: email_3f2a9b1c__attachment_1_Budget.xlsx",
        ["Budget", "3f2a9b1c__"],
    ),
]

# File names with spaces however they are logged, IC and phone numbers next
# to "_", and URL passwords holding "/" or "@" (FR-72).
CORPUS += [
    (
        "File processing error: - ID: doc-0123 Surat Tawaran Ali Bin Abu.docx",
        ["Surat", "Tawaran", "Ali Bin"],
    ),
    ("Upload failed for Laporan Prestasi Siti Aminah.pdf", ["Prestasi", "Siti"]),
    ("read /app/inputs/Ali Bin Abu/Laporan Prestasi.pdf", ["Ali Bin", "Prestasi"]),
    ("open C:\\Users\\Ali Bin Abu\\Laporan Kewangan 2024.docx", ["Ali", "Kewangan"]),
    ("Docling: /app/inputs/ws/Gaji Ali (1).pdf", ["Gaji Ali", "(1)"]),
    ("Docling: /app/inputs/ws/Gaji Ali bin Abu [final].docx", ["Gaji Ali", "bin Abu"]),
    ('OSError: "/x/inputs/ws/Ali\'s Gaji Oktober.pdf"', ["Ali's", "Gaji", "Oktober"]),
    (
        "/app/inputs/ws/Minit Mesyuarat Jabatan Kewangan Bil 3 Tahun 2024.pdf",
        ["Mesyuarat", "Kewangan", "Tahun"],
    ),
    ("/app/inputs/ws/Gaji, Ali bin Abu.docx", ["Gaji,", "Ali bin"]),
    ("nric_900101-13-5678 bad, kp 900101-13-5678_baru", ["900101-13-5678"]),
    ("[Borang_900101-13-5678] phone_0123456789", ["900101-13-5678", "0123456789"]),
    ("connect postgresql://lightrag:pa/ss0rd@db:5432/x failed", ["pa/ss0rd", "ss0rd"]),
    ("neo4j://neo4j:p@ss0rd@neo4j:7687 unreachable", ["ss0rd", "neo4j:p"]),
]

# Identifiers that must come through unchanged.
MUST_SURVIVE = [
    "doc-33e75ff09dd601bbe69f351039152189",
    "chunk-6512bd43d9caa6e02c990b0a82652dca",
    "request 0f8d2c1e9a7b4c3d8e6f5a4b3c2d1e0f",
    "5f2b0c9e-4d7a-4f3b-9c1e-0a8b7c6d5e4f",
    'File "/app/.venv/lib/python3.12/site-packages/starlette/Routing.py", line 7',
    "max_tokens=500 token_count=3 mytoken=abc",
]


@pytest.mark.parametrize("text,forbidden", CORPUS)
def test_corpus_nothing_survives(text, forbidden):
    out = redact(text)
    for needle in forbidden:
        assert needle not in out, (needle, out)


@pytest.mark.parametrize("text,_", CORPUS)
def test_redact_is_idempotent(text, _):
    once = redact(text)
    assert redact(once) == once


@pytest.mark.parametrize("text", MUST_SURVIVE)
def test_identifiers_survive(text):
    assert redact(text) == text


def test_hex_ids_never_lose_digit_runs():
    """No doc-<md5> or uuid4 hex ID is rewritten, whatever its digits."""
    import hashlib
    import uuid

    for i in range(5000):
        doc_id = "doc-" + hashlib.md5(str(i).encode()).hexdigest()
        request_id = uuid.uuid4().hex
        line = f"Processing {doc_id} (request {request_id})"
        assert redact(line) == line


# Inputs a client controls (an access-log path) or a traceback can carry.
# Each must redact in a few milliseconds: the patterns do bounded work per
# start position, so the cost is linear in the length.
_PATHOLOGICAL = [
    "/" + "a" * 8000 + "@",
    "/" + "a-" * 4000 + "?token=x",
    "a." * 4000 + "@",
    "postgres://" + "a" * 8000 + "@",
    "/" + "token=" * 1500,
    "password='" + "a" * 8000,
    "x://" * 2000 + "@",
    "Authorization:" * 600,
    "'" + "a b " * 2000,
]


@pytest.mark.parametrize("text", _PATHOLOGICAL)
def test_pathological_input_has_bounded_cost(text):
    for fn, payload in ((redact, text), (redact_traceback, text * 4)):
        fn(payload)  # warm up
        best = min(_elapsed(fn, payload) for _ in range(3))
        # The quadratic versions took 50 ms to 1.3 s; linear ones take ~1 ms.
        assert best < 0.025, (fn.__name__, text[:30], best)


def _elapsed(fn, payload) -> float:
    start = time.perf_counter()
    fn(payload)
    return time.perf_counter() - start


def test_access_log_record_with_hostile_path_is_cheap():
    rec = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("10.0.0.1:5000", "GET", "/" + "a-" * 3000 + "?token=1", "1.1", 404),
        None,
    )
    start = time.perf_counter()
    RedactFilter().filter(rec)
    assert time.perf_counter() - start < 0.025
    assert len(rec.args[2]) <= MSG_MAX_CHARS and "token=1" not in rec.args[2]


def test_ordinary_text_is_left_alone():
    text = (
        "[opso_unimas_my] Processing d-id: doc-0123abcdef0123abcdef0123abcdef01 "
        "max_tokens=500 token_count=3 batch 3/12 took 12.5s at 2026-11-03 10:12:13 "
        'File "/app/.venv/lib/python3.12/site-packages/starlette/routing.py", line 7'
    )
    assert redact(text) == text


def test_caps_and_newlines():
    long = "x" * (MSG_MAX_CHARS * 3)
    out = redact(long)
    assert len(out) <= MSG_MAX_CHARS
    assert out.endswith("chars]")
    assert redact(out) == out
    assert redact("a\r\nb\rc") == "a\nb\nc"
    assert redact(None) == ""


def test_traceback_keeps_head_and_last_frames():
    frames = "".join(
        f'  File "/app/lightrag/mod{i}.py", line {i}, in f{i}\n    call()\n'
        for i in range(600)
    )
    trace = "Traceback (most recent call last):\n" + frames + "ValueError: boom"
    out = redact_traceback(trace)
    assert len(out) <= EXC_MAX_CHARS
    assert out.startswith("Traceback (most recent call last):")
    assert out.endswith("ValueError: boom")
    assert "chars omitted" in out
    assert redact_traceback(out) == out


def test_fingerprint_stable_across_fstring_variants():
    a = normalise_message("[ws_one] Failed doc-0123abcdef after 3 tries: 'a.pdf'")
    b = normalise_message("[ws_two] Failed doc-99999999aa after 12 tries: 'b.pdf'")
    assert a == b
    fp_a = fingerprint("lightrag", "pipeline", 812, None, logging.WARNING, a)
    fp_b = fingerprint("lightrag", "pipeline", 812, None, "WARNING", b)
    assert fp_a == fp_b
    assert len(fp_a) == 16
    # The call site, not the message, is the identity.
    assert fingerprint("lightrag", "pipeline", 812, None, "WARNING") != fingerprint(
        "lightrag", "pipeline", 813, None, "WARNING"
    )
    assert fingerprint("lightrag", "pipeline", 812, "KeyError", "ERROR") != (
        fingerprint("lightrag", "pipeline", 812, "ValueError", "ERROR")
    )


def test_mask_path_and_text_ref():
    assert mask_path("/app/data/inputs/ws/Senate minutes, September.pdf") == (
        "Sen•••er.pdf"
    )
    assert mask_path("email_3f2a9b1c__master.txt") == "email_3f2a__master"
    assert mask_path("email_3f2a9b1c__attachment_2_Report.pdf.txt") == (
        "email_3f2a__attachment_Rep•••rt.pdf"
    )
    assert mask_path(None) == "-"
    assert mask_path(mask_path("Senate_minutes.pdf")) == mask_path("Senate_minutes.pdf")
    ref = text_ref("Quarterly budget review")
    assert ref.startswith("len=23 sha=") and "budget" not in ref


def test_sql_params_are_described_not_logged():
    described = describe_sql_params(
        {"workspace": "opso_unimas_my", "ids": ["a", "b"], "n": 3, "x": None}
    )
    assert described == "{workspace: str/14, ids: list/2, n: int, x: NoneType}"
    assert "opso" not in described
    assert describe_sql_params(None) == "{}"
    assert describe_sql_params(["secret text"]) == "[str/11]"


def _median_us(text: str, runs: int = 2000) -> float:
    for _ in range(200):
        redact(text)
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        redact(text)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e6


_FILLER = (
    " the upstream returned an error and the step will be retried with "
    "backoff according to the configured policy for this workspace"
) * 6

# Typical 500-character lines: at most one path, secret or address each.
_TYPICAL_LINES = {
    "plain": (
        "[opso_unimas_my] Processing batch 3/12 of documents, extracting "
        "entities and relations from chunk-abc "
    )
    * 6,
    "numbers": (
        "Completed 1234 of 5678 docs in 12.5s for workspace opso_unimas_my at "
        "2026-11-03 10:12:13 "
    )
    * 6,
    "one_path": "[opso_unimas_my] Failed to extract document 3/12: "
    "/app/data/inputs/opso_unimas_my/Laporan_Kewangan_2024.pdf" + _FILLER,
    "one_secret": "[opso_unimas_my] provider call failed with api_key=abc123secret"
    + _FILLER,
    "one_email": "Skipped LightRAG's own email from alert@unimas.my" + _FILLER,
}

# Lines dense with rule hits: five paths, nine secrets or ten key-like runs in
# 500 characters. Rare in practice; they get twice the typical budget.
_DENSE_LINES = {
    "paths": (
        "Failed to extract document 3/12: /app/data/inputs/opso_unimas_my/"
        "Laporan_Kewangan_2024.pdf after 3 retries; "
    )
    * 6,
    "secrets": (
        "provider call with api_key=abc123secret and Authorization: Bearer "
        "abcdef1234567890 failed for token=xyz "
    )
    * 6,
    "long_runs": (
        "chunk-3f2a9b1c3f2a9b1c3f2a9b1c3f2a9b1c3f2a9b1c3f2a9b1c and "
        "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 "
    )
    * 6,
}


@pytest.mark.parametrize("name", sorted(_TYPICAL_LINES))
def test_redaction_budget_typical_500_char_line(name):
    """Target (§3.8.4): under 20 us per typical 500-character line, median.

    The assertion allows 3x headroom so a slow or busy machine does not
    flake; a regression to a per-token or backtracking rule costs far more.
    """
    assert _median_us(_TYPICAL_LINES[name][:500]) < 60.0


@pytest.mark.parametrize("name", sorted(_DENSE_LINES))
def test_redaction_budget_dense_500_char_line(name):
    """Target: under 40 us per rule-dense 500-character line, median (3x)."""
    assert _median_us(_DENSE_LINES[name][:500]) < 120.0


def _record(msg, *args, exc_info=None, name="lightrag"):
    return logging.LogRecord(name, logging.ERROR, __file__, 10, msg, args, exc_info)


def test_redact_filter_rewrites_record_once():
    rec = _record("login for %s with token=%s", "alice@unimas.my", "abc123")
    flt = RedactFilter()
    assert flt.filter(rec) is True
    assert rec.getMessage() == "login for <email> with token=<redacted>"
    assert rec.args is None
    # Second handler: nothing changes, nothing re-runs.
    assert flt.filter(rec) is True
    assert rec.getMessage() == "login for <email> with token=<redacted>"


def test_redact_filter_keeps_access_log_args_shape():
    rec = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("10.0.0.1:5000", "GET", "/graph/search?token=abc&q=x", "1.1", 200),
        None,
    )
    RedactFilter().filter(rec)
    assert isinstance(rec.args, tuple) and len(rec.args) == 5
    assert rec.args[2] == "/graph/search?token=<redacted>&q=x"


def test_redact_filter_redacts_tracebacks():
    try:
        raise RuntimeError("failed with key sk-test-ABCDEFGH12345678")
    except RuntimeError:
        import sys

        rec = _record("boom", exc_info=sys.exc_info())
    RedactFilter().filter(rec)
    assert "sk-test-ABCDEFGH12345678" not in rec.exc_text
    assert "RuntimeError" in rec.exc_text
    formatted = logging.Formatter().format(rec)
    assert "ABCDEFGH12345678" not in formatted


def test_traceback_logged_as_message_keeps_its_exception_line():
    """logger.error(traceback.format_exc()) keeps head and last frames."""
    frames = "".join(
        f'  File "/app/lightrag/pipeline.py", line {i}, in step_{i}\n'
        f"    await self.stage_{i}(ctx)\n"
        for i in range(60)
    )
    trace = (
        "Traceback (most recent call last):\n"
        + frames
        + "ValueError: the real cause is here"
    )
    assert len(trace) > MSG_MAX_CHARS
    for msg, args in ((trace, ()), ("[ws] Traceback: %s", (trace,))):
        rec = _record(msg, *args)
        RedactFilter().filter(rec)
        out = rec.getMessage()
        assert out.endswith("ValueError: the real cause is here"), out[-80:]
        assert len(out) <= EXC_MAX_CHARS


def test_file_and_console_handler_output_is_redacted(tmp_path, capsys):
    """setup_logger's file and console handlers carry the redaction filter."""
    from lightrag.utils import setup_logger

    name = "lightrag_ops_redact_test"
    log_file = tmp_path / "logs" / "test.log"
    setup_logger(name, "INFO", log_file_path=str(log_file))
    target = logging.getLogger(name)
    try:
        target.warning(
            "Duplicate document detected: doc-1 (Gaji_Ahmad.pdf) from "
            "ahmad@unimas.my IC 900101-13-5678 api_key=sk-live-ABCDEFGH1234"
        )
        for handler in target.handlers:
            handler.flush()
        written = log_file.read_text(encoding="utf-8")
        console = capsys.readouterr().err
        for output in (written, console):
            assert "ahmad@unimas.my" not in output
            assert "900101-13-5678" not in output
            assert "sk-live-ABCDEFGH1234" not in output
            assert "Gaji_Ahmad" not in output
            assert "<email>" in output
        # The correlation fields are part of the file format.
        assert "] [-] [-] - Duplicate document detected" in written
        assert oct(log_file.parent.stat().st_mode & 0o777) == "0o700"
    finally:
        for handler in list(target.handlers):
            handler.close()
            target.removeHandler(handler)


# -- legacy pipeline text shown by the Operations console (D9) -----------------------

_PIPELINE_CORPUS = [
    (
        "Failed to extract document 1/3: Minit Mesyuarat Jabatan Keselamatan.pdf",
        "Failed to extract document 1/3: Min•••an.pdf",
    ),
    (
        "Merging stage failed in document 2/3: inputs/a/Surat Tawaran Ali bin Abu.pdf",
        "Merging stage failed in document 2/3: Sur•••bu.pdf",
    ),
    (
        "User cancelled during merge 1/4: inputs/user1/Minit Mesyuarat Jabatan.docx",
        "User cancelled during merge 1/4: Min•••an.docx",
    ),
    (
        "Completed processing file 3/10: senarai_pelajar_ahmad_bin_ali.json",
        "Completed processing file 3/10: sen•••li.json",
    ),
    ("Merging stage 4/9: config dump.xml", "Merging stage 4/9: con•••mp.xml"),
    (
        "Duplicate content skipped after parsing: Laporan Kewangan Ali bin Abu "
        "2024.docx; archived to /srv/archive/Laporan Kewangan Ali bin Abu 2024.docx",
        "Duplicate content skipped after parsing: Lap•••.docx; archived to Lap•••.docx",
    ),
    (
        "Deleting doc-0a1b2c3d Surat Tawaran Ali bin Abu.pdf(previous status: FAILED)",
        "Deleting doc-0a1b2c3d Sur•••bu.pdf(previous status: FAILED)",
    ),
    (
        "Error deleting document 2/5: doc-0a1b[Big Report (final).xml] - not found",
        "Error deleting document 2/5: doc-0a1b[Big•••l).xml] - not found",
    ),
    (
        "Skipping already processed file: data export 2024.json",
        "Skipping already processed file: dat•••.json",
    ),
    (
        "Failed to delete file Big Report.pdf: Permission denied",
        "Failed to delete file Big•••rt.pdf: Permission denied",
    ),
    (
        "Successfully deleted source files: a b c.pdf, d e.json",
        "Successfully deleted source files: a•••.pdf, d•••.json",
    ),
    # Lines that are not about a file stay as they were.
    (
        "Processed entity batch 1/3: 500 records (total: 10/20)",
        "Processed entity batch 1/3: 500 records (total: 10/20)",
    ),
    ("Deleting document 1/3: doc-0a1b", "Deleting document 1/3: doc-0a1b"),
]


@pytest.mark.parametrize("raw,expected", _PIPELINE_CORPUS)
def test_pipeline_lines_mask_the_whole_file_name(raw, expected):
    assert redact_pipeline(raw) == expected
    assert redact_pipeline(expected) == expected


def test_message_words_and_ids_end_a_file_name():
    """The name is masked whole, the message around it is kept."""
    assert (
        redact("File processing error: - ID: doc-0123 Surat Tawaran Ali.docx")
        == "File processing error: - ID: doc-0123 Sur•••li.docx"
    )
    assert redact("Upload failed for Laporan Prestasi.pdf") == (
        "Upload failed for Lap•••si.pdf"
    )
    assert redact("Laporan Prestasi.pdf could not be parsed") == (
        "Lap•••si.pdf could not be parsed"
    )
    assert redact("Processing .eml file: x") == "Processing .eml file: x"
    assert redact("can't open 'Report.pdf'") == "can't open 'Rep•••rt.pdf'"


def test_redact_filter_masks_deletion_lines_of_any_extension():
    """Stored file_path values in known shapes are masked whole on every
    handler, also when the type is not a document type (.json)."""
    for raw, gone in (
        ("File deletion skipped, missing or unsafe file: Gaji Ali.json", "Gaji"),
        ("Error deleting document 1/2: doc-1[senarai_ahmad_bin_ali.json] - x", "ahmad"),
    ):
        rec = _record(raw)
        RedactFilter().filter(rec)
        assert gone not in rec.getMessage(), rec.getMessage()


def test_pipeline_text_masks_a_known_path_anywhere():
    path = "inputs/u/Minit Mesyuarat Jabatan.json"
    text = "ValueError: could not parse Minit Mesyuarat Jabatan.json (row 3)"
    assert (
        redact_pipeline(text, paths=(path, None))
        == "ValueError: could not parse Min•••an.json (row 3)"
    )
    # Multi-line text: each line on its own, other rules still apply.
    two = "Extracting stage 1/2: My Notes.txt\nsent to someone@unimas.my"
    out = redact_pipeline(two)
    assert out.startswith("Extracting stage 1/2: My •••es.txt\n")
    assert "someone@unimas.my" not in out
    assert redact_pipeline(None) == ""
    assert len(redact_pipeline("Merging stage 1/2: " + "x" * 5000, 100)) <= 100


def test_reveal_keeps_file_names_but_every_other_rule():
    """mask_paths=False (the audited reveal only) lifts the file-name masking;
    secrets, emails, IC and phone numbers stay masked."""
    from lightrag.api.ops.redact import redact

    text = (
        "Senate minutes 900101-13-5678.pdf for someone.private@unimas.my "
        "key sk-test-SECRET1234567890abcdef call 012-345 6789"
    )
    masked = redact(text)
    revealed = redact(text, mask_paths=False)
    assert ".pdf" not in masked
    assert revealed.startswith("Senate minutes <ic>.pdf")
    for secret in ("900101", "someone.private", "SECRET1234", "345 6789"):
        assert secret not in revealed
    assert redact("report_2024.pdf", mask_paths=False) == "report_2024.pdf"
