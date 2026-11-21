"""Synthetic actual CLI content-integrity and legacy recovery regressions."""
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(os.environ.get('HTML_SOURCE', str(Path(__file__).resolve().parents[1])))


def run_cli(tmp_path, assigned):
    source = tmp_path / 'input.html'
    output = tmp_path / 'output.json'
    source.write_text('<script>var jsonData = ' + assigned + ';</script>')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SOURCE / 'src')
    result = subprocess.run(
        [sys.executable, '-m', 'conversation_analysis.ingest.html_extract',
         '--input', str(source), '--output', str(output)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    return result, output


@pytest.mark.parametrize('text', [
    'hello', 'literal ,} token', 'literal &amp; token', 'literal &lt;token&gt;',
    'literal &amp; and &quot;', 'literal ] bracket', 'literal [ bracket',
    'literal [x] bracket', 'escaped " quote and \\ slash with ] and [',
])
def test_valid_literal_json_round_trips(tmp_path, text):
    payload = [{'text': text}]
    result, output = run_cli(tmp_path, json.dumps(payload))
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == payload
    assert 'legacy' not in result.stderr.lower()


@pytest.mark.parametrize('legacy', ['entities', 'commas', 'combined'])
def test_legacy_recovery_does_not_change_literal_string_commas(tmp_path, legacy):
    payload = [{'text': 'literal ,} and [x]'}]
    assigned = json.dumps(payload)
    if legacy in ('commas', 'combined'):
        assigned = assigned[:-2] + ',},]'
    if legacy in ('entities', 'combined'):
        assigned = html.escape(assigned, quote=True)
    result, output = run_cli(tmp_path, assigned)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == payload
    assert 'legacy' in result.stderr.lower()


@pytest.mark.parametrize('assigned', ['not-json', '[{"text":broken}]', '[{"text":"unterminated]'])
def test_invalid_extraction_fails_cli(tmp_path, assigned):
    result, output = run_cli(tmp_path, assigned)
    assert result.returncode != 0
    assert not output.exists()
