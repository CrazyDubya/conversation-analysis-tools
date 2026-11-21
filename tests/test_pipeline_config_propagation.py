"""Documented YAML fields reach the real root CLI and existing analysis primitives."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(os.environ.get('ANALYSIS_SOURCE', Path(__file__).resolve().parents[1]))
TEXT = ('Research algorithms compare synthetic training examples. '
        'Research models evaluate reproducible training methods. '
        'Research findings explain the training observations. '
        'Research results support controlled algorithm evaluation. '
        'Research notes preserve the synthetic measurements.')
CONTROL = '''
import json,sys
sys.path.insert(0,sys.argv[1])
from conversation_analysis.pipeline.relevance_scorer import RelevanceScorer
from conversation_analysis.pipeline.summarizer import ExtractiveSummarizer
config=json.loads(sys.argv[2]);text=sys.argv[3]
weights=config.get('relevance',{}).get('weights')
sentences=config.get('summarizer',{}).get('summary_sentences',config.get('summary_sentences',3))
print(json.dumps({'relevance':RelevanceScorer(config.get('keywords',[])).rank_documents([(1,text)],weights)[0][2],
 'summary':ExtractiveSummarizer().summarize_to_text(text,sentences)}))
'''


class PipelineConfigurationCLI(unittest.TestCase):
    def result(self, config):
        with tempfile.TemporaryDirectory(prefix='analysis-config-test-') as temp:
            cwd = Path(temp)
            (cwd / 'logs').mkdir()  # This outcome is independent of fresh CLI startup.
            guard = cwd / 'guard'
            guard.mkdir()
            (guard / 'sitecustomize.py').write_text(
                'import socket, pathlib, os\n'
                'def blocked(*a, **k):\n'
                '    pathlib.Path(os.environ["GUARD_ATTEMPTS"]).write_text("attempted")\n'
                '    raise AssertionError("Network prohibited")\n'
                'socket.socket.connect = blocked\n'
                'socket.create_connection = blocked\n')
            with sqlite3.connect(cwd / 'fixture.db') as c:
                c.executescript('CREATE TABLE conversations(id INTEGER PRIMARY KEY,platform TEXT,title TEXT);'
                                'CREATE TABLE messages(id INTEGER PRIMARY KEY,conversation_id INTEGER,sender TEXT,content TEXT);')
                c.execute('INSERT INTO conversations VALUES(1,?,?)', ('claude', 'Synthetic'))
                c.execute('INSERT INTO messages VALUES(1,1,?,?)', ('assistant', TEXT))
            config_path = cwd / 'custom.yaml'
            # JSON is a YAML subset and PyYAML is used by the actual root CLI.
            config_path.write_text(json.dumps(config))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(guard), str(ROOT / 'src')]),
                       GUARD_ATTEMPTS=str(cwd / 'network-attempts'), HOME=str(cwd), PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, str(ROOT / 'run_pipeline.py'),
                                     '--config', str(config_path), '--db', str(cwd / 'fixture.db'),
                                     '--output', str(cwd / 'result.json'), '--no-save-db'],
                                    cwd=cwd, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((cwd / 'network-attempts').exists())
            data = json.loads((cwd / 'result.json').read_text())
            self.assertEqual(data['metadata']['total_documents'], 1)
            control = subprocess.run([sys.executable, '-c', CONTROL, str(ROOT / 'src'),
                                      json.dumps(config), TEXT], cwd=cwd, env=env,
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(control.returncode, 0, control.stdout + control.stderr)
            self.assertFalse((cwd / 'network-attempts').exists())
            return data, json.loads(control.stdout)

    def test_relevance_weights_match_existing_scorer(self):
        scores = []
        for weights in [{'density': 1, 'coverage': 0, 'tfidf': 0},
                        {'density': 0, 'coverage': 1, 'tfidf': 0}]:
            data, expected = self.result({'keywords': ['research'], 'relevance': {'weights': weights}})
            actual = data['documents']['1']['relevance']
            self.assertEqual(actual, expected['relevance'])
            self.assertEqual(data['statistics']['relevance']['average'], actual['combined'])
            self.assertEqual(data['documents']['1']['priority']['metadata']['relevance_score'], actual['combined'])
            scores.append(actual['combined'])
        self.assertNotEqual(*scores)

    def test_nested_summary_sentences_match_existing_summarizer(self):
        summaries = []
        for count in [1, 4]:
            data, expected = self.result({'keywords': ['research'], 'summarizer': {'summary_sentences': count}})
            self.assertEqual(data['documents']['1']['summary'], expected['summary'])
            summaries.append(expected['summary'])
        self.assertNotEqual(*summaries)

    def test_default_analysis_values_stay_unchanged(self):
        data, expected = self.result({'keywords': ['research']})
        self.assertEqual(data['documents']['1']['summary'], expected['summary'])
        self.assertEqual(data['documents']['1']['relevance'], expected['relevance'])

    def test_legacy_top_level_summary_fallback_and_nested_precedence(self):
        for config in [{'keywords': ['research'], 'summary_sentences': 2},
                       {'keywords': ['research'], 'summary_sentences': 2,
                        'summarizer': {'summary_sentences': 4}}]:
            data, expected = self.result(config)
            self.assertEqual(data['documents']['1']['summary'], expected['summary'])
