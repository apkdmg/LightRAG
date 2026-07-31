"""Offline tests for the URL/identifier-blob entity-name guard.

Covers the `_reject_invalid_entity_name` helper, both extraction parsers
(JSON and delimiter formats — the complete funnel for fresh, gleaning, cached,
and rebuild flows), and the relation-chunks key-width invariant that motivated
the guard (composite `src<SEP>tgt` key overflowing VARCHAR(512)).
"""

import json

import pytest

from lightrag.constants import DEFAULT_ENTITY_NAME_MAX_LENGTH, GRAPH_FIELD_SEP
from lightrag.operate import (
    _process_extraction_result,
    _process_json_extraction_result,
    _reject_invalid_entity_name,
)
from lightrag.utils import make_relation_chunk_key

pytestmark = pytest.mark.offline

TUPLE = "<|#|>"
COMPLETE = "<|COMPLETE|>"

# Realistic junk from the failing prod document: a URL-encoded tracking token
# and a full click-tracking URL, both capped at 256 chars by truncation.
TRACKING_BLOB = (
    "CYfQN-2BDJPhdJGWrkLYz-2FlaiTnuj6Vl3TpLxLopAky8uWtPblX6tSJy19-2BxKUaV3"
    "qmeCZ4T5gPKTgFuxbIbkO6T1vzez7w-2Ff-2BorEwqNwGFtSMhHRgiZR4rmD3v6DX9xhu"
    "RjACa1VGxgyvK7GGrHbLCn-2BezXz1wCFD-2FeeVvdob-2BQcen5DHGRJeaspAyzhbPfl"
    "u8sJQIamaO71MWMcgQxxR7Fj8-2BQCU8i1t7k53NzaBxwl025"
)
TRACKING_URL = (
    "https://ablink.students.udemy.com/ls/click?upn=u001.MPWQNoKfOWucilOG4GS8"
    "lguLLBbPRONAPVNgdA4KZXBKmJNUhucnxrylEzapu1H0757Wbq-2BcLevPwm0iWQEj-2BUuX"
)


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #


def test_rejects_urls():
    for name in (
        "https://example.com/page",
        "http://example.com",
        "HTTPS://EXAMPLE.COM",
        "www.example.com",
        "ftp://files.example.com",  # any scheme via ://
        TRACKING_URL,
    ):
        assert _reject_invalid_entity_name(name, "chunk-1", "Entity name"), name


def test_rejects_identifier_blobs():
    assert _reject_invalid_entity_name(TRACKING_BLOB, "chunk-1", "Entity name")
    assert _reject_invalid_entity_name("x" * 100, "chunk-1", "Entity name")


def test_accepts_real_entity_names():
    for name in (
        "Universiti Malaysia Sarawak",
        "unimas_all@unimas.my",  # emails are legit entities in this graph
        "Hospital Pengajar dan Pusat Perubatan UNIMAS",
        "ablink.students.udemy.com",  # bare domain: short, no scheme
        "100 Days of Code™: The Complete Python Pro Bootcamp",
        "AI Coder: Complete Claude Code & Coding Agents Course",  # colon but no "://"
        "沙捞越大学",  # CJK
        "x" * 99,  # just below the blob threshold
        # long but contains whitespace → not a blob
        "Mesyuarat Jawatankuasa Akaun Enterprise Bilangan Dua Tahun Dua Ribu "
        "Dua Puluh Enam Kali Kelima Belas Di Bilik Mesyuarat Utama Canselori",
    ):
        assert not _reject_invalid_entity_name(name, "chunk-1", "Entity name"), name


# --------------------------------------------------------------------------- #
# Text (delimiter) parser                                                     #
# --------------------------------------------------------------------------- #


def _entity_record(name: str, etype: str = "Organization") -> str:
    return f"entity{TUPLE}{name}{TUPLE}{etype}{TUPLE}A description of {name[:30]}"


def _relation_record(src: str, tgt: str) -> str:
    return f"relation{TUPLE}{src}{TUPLE}{tgt}{TUPLE}links to{TUPLE}link"


async def test_text_parser_drops_url_entities_and_their_relations():
    result = "\n".join(
        [
            _entity_record("Udemy"),
            _entity_record(TRACKING_URL, "Content"),
            _entity_record(TRACKING_BLOB, "Data"),
            _relation_record("Udemy", "Universiti Malaysia Sarawak"),
            _relation_record(TRACKING_BLOB, TRACKING_URL),  # the crashing pair
            _relation_record("Udemy", TRACKING_URL),  # one junk endpoint
            COMPLETE,
        ]
    )
    maybe_nodes, maybe_edges = await _process_extraction_result(
        result, "chunk-1", 0, "test.txt"
    )

    assert "Udemy" in maybe_nodes
    assert not any("ablink" in n or n == TRACKING_BLOB for n in maybe_nodes)
    assert ("Udemy", "Universiti Malaysia Sarawak") in maybe_edges
    assert len(maybe_edges) == 1  # both junk relations dropped


# --------------------------------------------------------------------------- #
# JSON parser                                                                 #
# --------------------------------------------------------------------------- #


async def test_json_parser_drops_url_entities_and_their_relations():
    payload = {
        "entities": [
            {"name": "Udemy", "type": "Organization", "description": "Course site"},
            {"name": TRACKING_URL, "type": "Content", "description": "A link"},
            {"name": TRACKING_BLOB, "type": "Data", "description": "A token"},
        ],
        "relationships": [
            {
                "source": "Udemy",
                "target": "Universiti Malaysia Sarawak",
                "description": "offers courses to",
                "keywords": "education",
            },
            {
                "source": TRACKING_BLOB,
                "target": TRACKING_URL,
                "description": "redirects to",
                "keywords": "link",
            },
            {
                "source": "Udemy",
                "target": TRACKING_URL,
                "description": "links to",
                "keywords": "link",
            },
        ],
    }
    maybe_nodes, maybe_edges = await _process_json_extraction_result(
        json.dumps(payload), "chunk-1", 0, "test.txt"
    )

    assert "Udemy" in maybe_nodes
    assert not any("ablink" in n or n == TRACKING_BLOB for n in maybe_nodes)
    assert ("Udemy", "Universiti Malaysia Sarawak") in maybe_edges
    assert len(maybe_edges) == 1


# --------------------------------------------------------------------------- #
# Key-width invariant (Fix 1 arithmetic)                                      #
# --------------------------------------------------------------------------- #


def test_relation_chunk_key_fits_widened_column():
    """Two max-length names + separator must fit VARCHAR(1024).

    Guards against a future DEFAULT_ENTITY_NAME_MAX_LENGTH bump silently
    reintroducing the StringDataRightTruncationError this fix addressed.
    """
    worst_case = 2 * DEFAULT_ENTITY_NAME_MAX_LENGTH + len(GRAPH_FIELD_SEP)
    assert worst_case <= 1024, (
        f"relation-chunk composite key can reach {worst_case} chars; widen "
        "LIGHTRAG_ENTITY_CHUNKS/LIGHTRAG_RELATION_CHUNKS id columns "
        "(and their migration) before raising the name cap"
    )
    key = make_relation_chunk_key(
        "a" * DEFAULT_ENTITY_NAME_MAX_LENGTH, "b" * DEFAULT_ENTITY_NAME_MAX_LENGTH
    )
    assert len(key) == worst_case
