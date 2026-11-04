"""Irreversible redaction of log text, plus the call-site fingerprint.

Everything here is pure and stdlib-only, so core modules can import it. The
same :func:`redact` runs on the text log file and the console (through
:class:`RedactFilter`), on the future structured ops log sink, and on read for
legacy text such as ``history_messages`` or ``error_msg``.

Rules, in order (FR-72):

- ``data:…;base64,`` blobs
- ``Authorization:`` values, ``api_key=`` / ``token=`` / ``secret=`` /
  ``password=`` style assignments, ``Bearer`` tokens, JWTs and ``sk-`` keys
- userinfo in DSNs and URLs
- email addresses
- long base64 or hex runs (40+ characters)
- Malaysian IC numbers, Malaysian phone numbers, digit runs of 12 or more
- file names (document, mail and image extensions, and the email bundle
  shape), masked with :func:`lightrag.api.insights.analytics.mask_name`; a
  quoted name is masked whole, spaces included

Every rule does bounded work per start position, so a line costs time linear
in its length whatever a client puts in a request path. Budget (median, per
500-character line): 20 us for typical lines, 40 us for lines dense with rule
hits (several paths, secrets or key-like runs); tests/test_ops_redact.py
checks both with headroom.

Every rule's replacement is a fixed point of every rule, so
``redact(redact(x)) == redact(x)``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from string import hexdigits
from typing import Any, Iterable, Optional

from lightrag.api.insights.analytics import document_label, mask_name

MSG_MAX_CHARS = 2048
EXC_MAX_CHARS = 8192

# Inputs longer than this many times the cap are cut before the regexes run,
# so a pathological message cannot cost more than a bounded amount of CPU.
# The cut point is always outside the part that survives the final cap.
_PRECUT_FACTOR = 4

_REDACTED = "<redacted>"
_HEX = frozenset(hexdigits)

_DATA_URI = re.compile(r"data:[\w.+-]{1,64}/[\w.+-]{1,64};base64,[A-Za-z0-9+/=]+")

# Every pattern below does a bounded amount of work per start position (no
# unbounded quantifier is followed by anything that can fail and backtrack
# into it), so the cost of a line stays linear in its length. RedactFilter
# runs on access-log lines whose path is chosen by unauthenticated clients.

# The value after a secret-like name. A quoted value runs to its closing quote
# (spaces and commas included); an unquoted one runs to whitespace, a quote,
# ";", "&", "}" or "]", and through commas that are not followed by a space.
_SECRET_VALUE = (
    r"(?:(?P<q>\\?['\"])(?P=q)|(?P<oq>\\?['\"]).{1,512}?(?P=oq)"
    r"|(?P<uq>\\?['\"])?(?:(?:bearer|basic|token|digest)\s+)?"
    r"(?:[^\s'\",;&}\]]|,(?!\s))+)"
)

# "Authorization: Bearer abc", "authorization='Basic abc'"
_AUTHORIZATION = re.compile(
    r"(?i)(?<![A-Za-z0-9])(authorization\\?['\"]?\s*[:=]\s*)" + _SECRET_VALUE
)

# "api_key=…", "x-api-key: …", '"password": "…"', "OPENAI_API_KEY=…".
# The keyword must be the END of the name ("max_tokens=5" and "token_count=3"
# are left alone) and must not continue a word ("mytoken=" is left alone);
# whatever precedes it ("OPENAI_", "client_") stays in the text.
_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])((?:api[_-]?key|access[_-]?key|secret[_-]?key|"
    r"private[_-]?key|token|secret|passw(?:or)?d|passwd|pwd|credentials?)"
    r"\\?['\"]?\s*[:=]\s*)" + _SECRET_VALUE
)

_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*")
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")

# scheme://user:pass@host -> scheme://<redacted>@host
_DSN_USERINFO = re.compile(
    r"(?<![A-Za-z0-9])([a-zA-Z][a-zA-Z0-9+.-]{1,20}://)[^\s/@'\"]{1,256}@"
)

# RFC 5321 limits: local part 64, label 63. The lookbehind makes a match start
# only at the beginning of a run of local-part characters, so a long run
# before an "@" is scanned once, not once per position.
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@"
    r"[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,24}"
)

# Runs of 40+ base64/base64url/hex characters. The callback decides whether a
# run is a secret: hex-only, or a mix of upper case, lower case and digits.
# Lower-case paths and snake_case identifiers in tracebacks are kept.
_LONG_RUN = re.compile(r"(?<![A-Za-z0-9+/_=-])[A-Za-z0-9+/_-]{40,}={0,2}")

# The digit rules start with a digit (or "+") rather than a lookbehind, so the
# regex engine can skip straight to candidate positions. A letter or digit on
# either side means the digits are part of an identifier (a doc-<md5> ID, a
# uuid hex request ID) and are kept; "_" and "-" still count as separators.
_IC_NUMBER = re.compile(r"\d(?<!\w\d)\d{5}-?\d{2}-?\d{4}\b")
_PHONE = re.compile(
    r"(?:\+(?<![\w+]\+)6?0|6(?<![\w+]6)0|0(?<![\w+]0))"
    r"[ \t-]?1\d[ \t-]?\d{3,4}[ \t-]?\d{4}(?![0-9A-Za-z])"
)
_DIGIT_RUN = re.compile(r"\d(?<![0-9A-Za-z]\d)\d{11,}(?![0-9A-Za-z])")

# File names. Office, text, mail, markup, e-book and image types LightRAG
# ingests or email carries. Code and data types (.py, .js, .json, .xml) are
# left out: tracebacks are full of them ("routing.py", "response.json()").
_FILE_EXTENSIONS = frozenset(
    (
        "pdf doc docx xls xlsx ppt pptx odt ods odp rtf txt csv eml msg md mdx "
        "htm html epub tex png jpg jpeg gif bmp tif tiff webp heic"
    ).split()
)
_PATH_EXTENSIONS = "|".join(sorted(_FILE_EXTENSIONS, key=lambda e: (-len(e), e)))
_PATH_HINT = re.compile(
    rf"(?i)__(?:master|inline|attachment)|\.(?:{_PATH_EXTENSIONS})(?![A-Za-z0-9])"
)
# A quoted name may contain spaces ("Minit Mesyuarat Jabatan.pdf"): the whole
# quoted span is masked, not just its last word.
_QUOTED_FILE = re.compile(
    rf"(?i)(?P<q>['\"`])(?P<name>[^'\"`\n]{{1,255}}?\.(?:{_PATH_EXTENSIONS}))(?P=q)"
)
# Characters that end a file-name token.
_PATH_STOP = re.compile(r"[\s'\"`()<>\[\]{},;|=:]")
_EMAIL_BUNDLE = re.compile(r"email_[0-9a-f]{4,}__(?:master|inline|attachment)")
# What follows a run of path characters when the run is a path, not a key.
_EXTENSION_AFTER = re.compile(r"\.[A-Za-z0-9]{1,8}(?![A-Za-z0-9])")
_MASK_MARK = "•••"

_NEWLINES = re.compile(r"\r\n?")
_HAS_DIGIT_RUN = re.compile(r"\d\d\d\d")
_SIX_DIGITS = re.compile(r"\d\d\d\d\d\d")
# keyword -> the letters _ASSIGNMENT allows between it and the separator
_ASSIGNMENT_KEYWORDS = {
    "key": ("",),
    "token": ("",),
    "secret": ("",),
    "pass": ("word", "wd"),
    "pwd": ("",),
    "credential": ("", "s"),
}
_ASSIGNMENT_GAP = frozenset("\\'\" \t")

# Message normalisation for the fingerprint tie-break.
_NORMALISE = re.compile(
    r"'[^']*'|\"[^\"]*\"|`[^`]*`|<[a-z0-9]+>|\b[0-9a-fA-F]{8,}\b|\d+(?:\.\d+)?"
)
_LEADING_SCOPE = re.compile(r"^\[[^\]]{1,128}\]\s*")


def _long_run(match: re.Match[str]) -> str:
    run = match.group(0)
    core = run.rstrip("=")
    if all(ch in _HEX for ch in core):
        return "<hex>"
    after = _EXTENSION_AFTER.match(match.string, match.end())
    if after and ("/" in core or after.group(0)[1:].lower() in _FILE_EXTENSIONS):
        # "/app/inputs/ws/Report_2024.pdf", "3.12/site-packages/x/Mod.py",
        # "email_<hex>__attachment_1_Budget.xlsx": a path or a file name,
        # left to the file-name rule.
        return run
    has_digit = has_upper = has_lower = False
    for ch in core:
        if ch.isdigit():
            has_digit = True
        elif ch.isupper():
            has_upper = True
        elif ch.islower():
            has_lower = True
        if has_digit and has_upper and has_lower:
            return "<b64>"
    return run


def _is_file_token(token: str) -> bool:
    _, dot, extension = token.rpartition(".")
    if dot and extension.lower() in _FILE_EXTENSIONS:
        return True
    return "__" in token and _EMAIL_BUNDLE.search(token) is not None


def _mask_token(token: str) -> str:
    if _MASK_MARK in token:
        return token
    trailing = ""
    while token and token[-1] in ".!?":
        trailing = token[-1] + trailing
        token = token[:-1]
    if not token or not _is_file_token(token):
        return token + trailing
    base = token[max(token.rfind("/"), token.rfind("\\")) + 1 :] or token
    label = document_label("", base)
    if label["kind"] == "email":
        return f"email_{label['code']}__master" + trailing
    if label["kind"] in ("email_attachment", "email_image"):
        part = "attachment" if label["kind"] == "email_attachment" else "inline"
        name = label["label"]
        return (
            f"email_{label['code']}__{part}" + (f"_{name}" if name else "") + trailing
        )
    return mask_name(base) + trailing


def _mask_quoted_file(match: re.Match[str]) -> str:
    quote = match.group("q")
    return quote + mask_path(match.group("name")) + quote


def _mask_path_tokens(text: str) -> str:
    """Mask each whitespace-free token that holds a file-name hint.

    Only the neighbourhood of a hint is examined (the token edges are found
    with C-level searches, backwards through the reversed text), so a long
    line with one file name costs little more than the hint search itself.
    """
    out: list[str] = []
    last = 0
    size = len(text)
    reverse: Optional[str] = None
    for hint in _PATH_HINT.finditer(text):
        if hint.start() < last:
            continue
        if reverse is None:
            reverse = text[::-1]
        # text[i] is reverse[size - 1 - i]; search back from hint.start() - 1.
        stop = _PATH_STOP.search(reverse, size - hint.start())
        start = max(last, size - stop.start() if stop else 0)
        stop = _PATH_STOP.search(text, hint.end())
        end = stop.start() if stop else size
        out.append(text[last:start])
        out.append(_mask_token(text[start:end]))
        last = end
    if not out:
        return text
    out.append(text[last:])
    return "".join(out)


def _secret_value(match: re.Match[str]) -> str:
    """Keep the name and separator; replace the value, keeping its quotes."""
    quote = match.group("q") or match.group("oq") or match.group("uq") or ""
    return match.group(1) + quote + _REDACTED + quote


def _may_assign_secret(lowered: str) -> bool:
    """True when a secret-like name is followed by ``=`` or ``:``.

    A pre-check that keeps the expensive case-insensitive regex off ordinary
    lines (``max_tokens=500`` does not pass it).
    """
    size = len(lowered)
    for keyword, suffixes in _ASSIGNMENT_KEYWORDS.items():
        start = lowered.find(keyword)
        while start != -1:
            end = start + len(keyword)
            pos = end
            while pos < size and lowered[pos].isalpha():
                pos += 1
            if lowered[end:pos] not in suffixes:
                start = lowered.find(keyword, start + 1)
                continue
            while pos < size and lowered[pos] in _ASSIGNMENT_GAP:
                pos += 1
            if pos < size and lowered[pos] in ":=":
                return True
            start = lowered.find(keyword, start + 1)
    return False


def _apply_rules(text: str) -> str:
    lowered = text.lower()
    if ";base64," in lowered:
        text = _DATA_URI.sub("data:<redacted>", text)
    if "authorization" in lowered:
        text = _AUTHORIZATION.sub(_secret_value, text)
    if _may_assign_secret(lowered):
        text = _ASSIGNMENT.sub(_secret_value, text)
    if "bearer" in lowered:
        text = _BEARER.sub("Bearer " + _REDACTED, text)
    if "eyj" in lowered:
        text = _JWT.sub("<jwt>", text)
    if "sk-" in lowered:
        text = _SK_KEY.sub("sk-" + _REDACTED, text)
    if "@" in text:
        if "://" in text:
            text = _DSN_USERINFO.sub(r"\1" + _REDACTED + "@", text)
        text = _EMAIL.sub("<email>", text)
    # A 40-character run needs a whitespace-free token at least that long.
    if len(text) >= 40 and max(map(len, text.split()), default=0) >= 40:
        text = _LONG_RUN.sub(_long_run, text)
    # Every digit rule needs four consecutive digits; IC numbers and long
    # digit runs need six.
    if _HAS_DIGIT_RUN.search(text):
        if _SIX_DIGITS.search(text):
            text = _IC_NUMBER.sub("<ic>", text)
            text = _DIGIT_RUN.sub("<num>", text)
        text = _PHONE.sub("<phone>", text)
    if _PATH_HINT.search(text):
        if "'" in text or '"' in text or "`" in text:
            text = _QUOTED_FILE.sub(_mask_quoted_file, text)
        text = _mask_path_tokens(text)
    return text


def _cap_head(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    marker = f"…[+{len(text) - max_len} chars]"
    keep = max(0, max_len - len(marker))
    return text[:keep] + marker


def redact(text: Any, max_len: int = MSG_MAX_CHARS) -> str:
    """Redact one message and cap it at ``max_len`` characters (head kept)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    if len(text) > max_len * _PRECUT_FACTOR:
        text = text[: max_len * _PRECUT_FACTOR]
    if "\r" in text:
        text = _NEWLINES.sub("\n", text)
    return _cap_head(_apply_rules(text), max_len)


def redact_traceback(text: Any, max_len: int = EXC_MAX_CHARS) -> str:
    """Redact a traceback; when too long keep its head and its last frames."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    budget = max_len * _PRECUT_FACTOR
    if len(text) > budget:
        half = budget // 2
        text = text[:half] + "\n" + text[-half:]
    if "\r" in text:
        text = _NEWLINES.sub("\n", text)
    text = _apply_rules(text)
    if len(text) <= max_len:
        return text
    omitted = len(text) - max_len
    marker = f"\n…[{omitted} chars omitted]…\n"
    room = max(0, max_len - len(marker))
    head = room // 4
    tail = room - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def mask_path(path: Optional[str]) -> str:
    """A log-safe stand-in for a document path: the masked basename only."""
    if not path:
        return "-"
    base = re.split(r"[\\/]", str(path))[-1] or str(path)
    if _MASK_MARK in base:
        return base
    label = document_label("", base)
    if label["kind"] == "email":
        return f"email_{label['code']}__master"
    if label["kind"] in ("email_attachment", "email_image"):
        part = "attachment" if label["kind"] == "email_attachment" else "inline"
        name = label["label"]
        return f"email_{label['code']}__{part}" + (f"_{name}" if name else "")
    return mask_name(base)


# Pipeline status lines and error_msg values that embed a document's path
# (lightrag/pipeline.py, lightrag/lightrag.py, lightrag/operate.py and
# lightrag/api/routers/document_routes.py). The owner's own feed keeps the
# name; the Operations console masks the whole name, spaces and any extension
# included, which the token rules above cannot see. Each pattern runs on one
# line; its "path" and "path2" groups are masked, a "list" group is masked
# item by item.
_PIPELINE_PATH_LINES = (
    re.compile(
        r"(?:Extracting stage|(?:Completed p|P)rocessing file|Merging stage"
        r"(?: failed in document)?|Failed to extract document|User cancelled"
        r"(?: during merge)?) \d+/\d+: (?P<path>.+)"
    ),
    re.compile(
        r"(?:Document deleted|Failed to delete|Error deleting document) "
        r"\d+/\d+: [^\s\[]+\[(?P<path>.+?)\](?= - |$)"
    ),
    re.compile(
        r"Duplicate content skipped after parsing: (?P<path>.+?)"
        r"(?:; archived to (?P<path2>.+))?$"
    ),
    re.compile(r"Deleting \S+ (?P<path>.+?)\(previous status: "),
    re.compile(
        r"(?:File deletion skipped, missing or unsafe file|Skipping already "
        r"processed file|Unsafe file path skipped): (?P<path>.+)"
    ),
    re.compile(r"Failed to delete file (?P<path>.+?): "),
    re.compile(r"Successfully deleted source files: (?P<list>.+)"),
)
# Something every pattern above needs, to skip ordinary lines cheaply.
_PIPELINE_PATH_HINT = re.compile(r"\d/\d|Duplicate content|Deleting |file")
_KNOWN_PATH_MIN = 4


def _mask_pipeline_line(line: str) -> str:
    for pattern in _PIPELINE_PATH_LINES:
        match = pattern.search(line)
        if match is None:
            continue
        spans = []
        for name, value in match.groupdict().items():
            if not value:
                continue
            if name == "list":
                masked = ", ".join(mask_path(v) for v in value.split(", "))
            else:
                masked = mask_path(value)
            spans.append((match.start(name), match.end(name), masked))
        for start, end, masked in sorted(spans, reverse=True):
            line = line[:start] + masked + line[end:]
        return line
    return line


def mask_pipeline_paths(text: str, paths: Iterable[Optional[str]] = ()) -> str:
    """Mask document paths in pipeline text before :func:`redact` runs.

    ``paths`` are the paths this text is known to be about (a doc row's own
    ``file_path``); each occurrence of the path or its basename is masked
    wherever it appears. Lines in a known pipeline shape are masked whatever
    the name looks like.
    """
    for path in paths:
        if not path or not isinstance(path, str):
            continue
        masked = mask_path(path)
        base = re.split(r"[\\/]", path)[-1]
        for needle in dict.fromkeys((path, base)):
            if len(needle) >= _KNOWN_PATH_MIN and needle in text:
                text = text.replace(needle, masked)
    if not _PIPELINE_PATH_HINT.search(text):
        return text
    return "\n".join(_mask_pipeline_line(line) for line in text.split("\n"))


def redact_pipeline(
    text: Any, max_len: int = MSG_MAX_CHARS, paths: Iterable[Optional[str]] = ()
) -> str:
    """:func:`redact` for legacy pipeline text shown by the Operations console.

    ``history_messages``, ``latest_message`` and ``error_msg`` embed document
    paths in fixed shapes; those are masked whole (D9) before the token rules.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    if len(text) > max_len * _PRECUT_FACTOR:
        text = text[: max_len * _PRECUT_FACTOR]
    if "\r" in text:
        text = _NEWLINES.sub("\n", text)
    return redact(mask_pipeline_paths(text, paths), max_len)


def text_ref(text: Any) -> str:
    """Length and a short hash, for logging that a value existed without it."""
    value = "" if text is None else str(text)
    digest = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:8]
    return f"len={len(value)} sha={digest}"


def normalise_message(message: Any, max_len: int = 256) -> str:
    """Collapse the variable parts of a message so f-string variants match."""
    text = "" if message is None else str(message)
    text = _LEADING_SCOPE.sub("[?] ", text, count=1)
    text = _NORMALISE.sub("?", text)
    return text[:max_len]


def fingerprint(
    logger_name: str,
    module: str,
    lineno: int,
    exc_type: Optional[str],
    level: str | int,
    normalised_msg: Optional[str] = None,
) -> str:
    """Stable id of a log call site. The message is only a tie-break."""
    parts = [
        logger_name or "",
        module or "",
        str(lineno or 0),
        exc_type or "",
        logging.getLevelName(level) if isinstance(level, int) else str(level),
    ]
    if normalised_msg:
        parts.append(normalised_msg)
    return hashlib.sha1("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


_EXC_FORMATTER = logging.Formatter()
_TRACEBACK_MARK = "Traceback (most recent call last):"


class RedactFilter(logging.Filter):
    """Handler filter that redacts a record in place before it is formatted.

    Attach it to every handler that writes text somewhere (file, console, ops
    sink). The record is marked, so the work is done once per record however
    many handlers it reaches. A record is never dropped: if redaction itself
    fails, the message is replaced rather than written raw.
    """

    def __init__(
        self,
        name: str = "",
        msg_max: int = MSG_MAX_CHARS,
        exc_max: int = EXC_MAX_CHARS,
    ):
        super().__init__(name)
        self.msg_max = msg_max
        self.exc_max = exc_max

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_lightrag_redacted", False):
            return True
        try:
            self._redact(record)
        except Exception as exc:  # pragma: no cover - defensive
            record.msg = f"[log line withheld: redaction failed ({type(exc).__name__})]"
            record.args = None
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        record._lightrag_redacted = True
        return True

    def _redact(self, record: logging.LogRecord) -> None:
        args = record.args
        if args and isinstance(args, tuple) and record.name.endswith(".access"):
            # Access-log formatters unpack the args tuple, so keep its shape.
            record.args = tuple(
                redact(a, self.msg_max) if isinstance(a, str) else a for a in args
            )
        else:
            message = record.getMessage()
            if _TRACEBACK_MARK in message:
                # logger.error(traceback.format_exc()): keep the head and the
                # last frames, where the exception line is.
                redacted = redact_traceback(message, self.exc_max)
            else:
                redacted = redact(message, self.msg_max)
            if redacted != message or args:
                record.msg = redacted
                record.args = None
        if record.exc_info and record.exc_info[0] is not None and not record.exc_text:
            record.exc_text = redact_traceback(
                _EXC_FORMATTER.formatException(record.exc_info), self.exc_max
            )
        elif record.exc_text:
            record.exc_text = redact_traceback(record.exc_text, self.exc_max)
        if record.stack_info:
            record.stack_info = redact_traceback(record.stack_info, self.exc_max)
