"""Ingest-layer tests: archive_parser round-trip and html_extract fixture."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation_analysis.ingest import archive_parser  # noqa: E402
from conversation_analysis.ingest import html_extract  # noqa: E402

SAMPLE = [
    {
        "id": "conv-1",
        "title": "Hello World",
        "create_time": 1700000000,
        "mapping": {
            "aaa": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Hi there"]},
                }
            },
            "bbb": {
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["Hello!"]},
                }
            },
        },
    }
]


def test_parse_conversations():
    parsed = archive_parser.parse_conversations(SAMPLE)
    assert len(parsed) == 1
    conv = parsed[0]
    assert conv["conversation_id"] == "conv-1"
    assert conv["title"] == "Hello_World"
    assert conv["messages"][0]["author"] == "user"
    assert conv["messages"][0]["content"] == ["Hi there"]
    assert conv["messages"][1]["author"] == "assistant"
    assert conv["messages"][1]["content"] == ["Hello!"]


def test_sanitize_filename():
    assert archive_parser.sanitize_filename("Hello / World?") == "HelloWorld"
    assert archive_parser.sanitize_filename("") == "conversation"


def test_save_conversations_to_files():
    tmp = tempfile.mkdtemp()
    parsed = archive_parser.parse_conversations(SAMPLE)
    archive_parser.save_conversations_to_files(parsed, tmp)
    files = os.listdir(tmp)
    assert files == ["Hello_Worl.json"]
    body = json.load(open(os.path.join(tmp, files[0])))
    assert body["title"] == "Hello_World"
    assert len(body["messages"]) == 2


def test_html_extract_var_jsondata():
    payload = json.dumps(SAMPLE)
    html = "<html><body><script>var jsonData = " + payload + ";</script></body></html>"
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "page.html")
    open(src, "w").write(html)
    out_json = os.path.join(tmp, "data.json")
    html_extract.extract_json_with_balanced_brackets(src, out_json)
    written = json.load(open(out_json))
    assert written[0]["title"] == "Hello World"
