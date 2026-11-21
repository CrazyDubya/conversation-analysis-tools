"""Exercise built/installed CLI outside the checkout and the existing root runner."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(os.environ.get('ANALYSIS_SOURCE', Path(__file__).resolve().parents[1]))
TEXT = 'Research algorithms compare synthetic training examples. Research models evaluate reproducible training methods.'


class InstalledPipelineCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sandbox = tempfile.TemporaryDirectory(prefix='analysis-installed-test-')
        cls.root = Path(cls.sandbox.name)
        source = cls.root / 'source'
        shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns('.git', '__pycache__'))
        cls.site = cls.root / 'installed'
        cls.guard = cls.root / 'guard'
        cls.guard.mkdir()
        (cls.guard / 'sitecustomize.py').write_text(
            'import socket, pathlib, os\n'
            'def blocked(*a, **k):\n'
            '    pathlib.Path(os.environ["GUARD_ATTEMPTS"]).write_text("attempted")\n'
            '    raise AssertionError("Network prohibited in installed CLI test")\n'
            'socket.socket.connect = blocked\n'
            'socket.create_connection = blocked\n')
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
            '--disable-pip-version-check', '--no-index', '--target', str(cls.site), str(source)
        ], cwd=cls.root, text=True, capture_output=True,
            env=dict(os.environ, PYTHONPATH=str(cls.guard), GUARD_ATTEMPTS=str(cls.root / 'network-attempts')), timeout=90)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.command = str(cls.site / 'bin' / 'analyze-content')

    @classmethod
    def tearDownClass(cls):
        cls.sandbox.cleanup()

    def run_cli(self, args, root_runner=False, precreate_logs=False):
        temp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(temp.cleanup)
        cwd = Path(temp.name)
        db = cwd / 'fixture.db'
        with sqlite3.connect(db) as c:
            c.executescript('CREATE TABLE conversations(id INTEGER PRIMARY KEY,platform TEXT,title TEXT);'
                            'CREATE TABLE messages(id INTEGER PRIMARY KEY,conversation_id INTEGER,sender TEXT,content TEXT);')
            c.execute('INSERT INTO conversations VALUES(1,?,?)', ('claude', 'Synthetic fixture'))
            c.execute('INSERT INTO messages VALUES(1,1,?,?)', ('assistant', TEXT))
        if precreate_logs:
            (cwd / 'logs').mkdir()
        command = [sys.executable, str(ROOT / 'run_pipeline.py')] if root_runner else [self.command]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(self.guard), str(self.site)]),
                   PYTHONDONTWRITEBYTECODE='1', HOME=str(cwd), GUARD_ATTEMPTS=str(self.root / 'network-attempts'))
        result = subprocess.run(command + args, cwd=cwd, env=env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / 'network-attempts').exists())
        return cwd, result

    def test_installed_help_outside_checkout_needs_no_files(self):
        cwd, result = self.run_cli(['--help'])
        self.assertIn('--no-save-db', result.stdout)
        self.assertFalse((cwd / 'logs').exists())
        self.assertFalse((cwd / 'output').exists())

    def test_installed_packaged_defaults_create_log_and_output(self):
        cwd, result = self.run_cli(['--db', 'fixture.db', '--no-save-db'])
        data = json.loads((cwd / 'output/pipeline_results.json').read_text())
        self.assertEqual(data['metadata']['total_documents'], 1)
        self.assertEqual(data['documents']['1']['text'], TEXT)
        self.assertTrue((cwd / 'logs/pipeline.log').is_file())
        self.assertIn(str(self.site), result.stdout)
        with sqlite3.connect(cwd / 'fixture.db') as c:
            self.assertEqual(c.execute("SELECT name FROM sqlite_master WHERE name='analysis_results'").fetchall(), [])

    def test_installed_explicit_config_and_custom_nested_output(self):
        # The source config is an explicit path; the default test above proves package data.
        cwd, _ = self.run_cli([
            '--config', str(ROOT / 'src/conversation_analysis/config/pipeline_config.yaml'),
            '--db', 'fixture.db', '--output', 'custom/deep/results.json',
            '--platform', 'claude', '--limit', '1'])
        data = json.loads((cwd / 'custom/deep/results.json').read_text())
        self.assertEqual(data['metadata']['total_documents'], 1)
        with sqlite3.connect(cwd / 'fixture.db') as c:
            self.assertEqual(c.execute('SELECT count(*) FROM analysis_results').fetchone()[0], 1)
        self.assertTrue((cwd / 'logs/pipeline.log').exists())

    def test_existing_root_explicit_runner_remains_usable(self):
        cwd, _ = self.run_cli([
            '--config', str(ROOT / 'src/conversation_analysis/config/pipeline_config.yaml'),
            '--db', 'fixture.db', '--output', 'root-results.json', '--no-save-db'],
            root_runner=True, precreate_logs=True)
        data = json.loads((cwd / 'root-results.json').read_text())
        self.assertEqual(data['documents']['1']['text'], TEXT)
        self.assertEqual(data['metadata']['total_documents'], 1)
