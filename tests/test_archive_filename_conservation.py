"""Complete archive CLI regressions, synthetic input and temporary output only."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArchiveFilenameConservation(unittest.TestCase):
    def invoke(self, titles):
        temp = tempfile.TemporaryDirectory(prefix='archive-filenames-')
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        data = [
            {'id': f'conversation-{i}', 'title': title, 'mapping': {
                'entry': {'id': f'message-{i}', 'message': {
                    'author': {'role': 'user'},
                    'content': {'parts': [f'Unique source message {i}']}
                }}
            }} for i, title in enumerate(titles)
        ]
        source = root / 'conversations.json'
        source.write_text(json.dumps(data))
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
        def run():
            result = subprocess.run(
                [sys.executable, '-m', 'conversation_analysis.ingest.archive_parser'],
                input=str(source) + '\n', cwd=root, env=env,
                capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            files = {p.name: json.loads(p.read_text()) for p in root.glob('gptlogs-*/*.json')}
            return files
        files = run()
        return source, data, files, run

    def conserved(self, titles):
        _, inputs, files, _ = self.invoke(titles)
        self.assertEqual(len(files), len(inputs), files)
        self.assertEqual({v['conversation_id'] for v in files.values()}, {v['id'] for v in inputs})
        for value in files.values():
            i = int(value['conversation_id'].split('-')[-1])
            self.assertEqual(value['messages'][0]['content'], [f'Unique source message {i}'])
        return files

    def test_literal_suffix_collision_in_each_order(self):
        for titles in sorted(set(itertools.permutations(['Foo', 'Foo', 'Foo_2']))):
            with self.subTest(titles=titles): self.conserved(titles)

    def test_sanitized_title_collision_with_literal_suffix(self):
        self.conserved(['A?', 'A!', 'A_2'])

    def test_truncated_title_collision_with_literal_suffix(self):
        self.conserved(['12345678??one', '12345678!!two', '12345678_2'])

    def test_collision_free_names_and_ordinary_duplicate_names_unchanged(self):
        files = self.conserved(['Foo', 'Foo', 'Bar'])
        self.assertEqual(set(files), {'Foo.json', 'Foo_2.json', 'Bar.json'})
        self.assertEqual(files['Foo.json']['conversation_id'], 'conversation-0')
        self.assertEqual(files['Foo_2.json']['conversation_id'], 'conversation-1')

    def test_existing_prior_run_overwrite_behavior_not_changed(self):
        source, data, files, run = self.invoke(['Ordinary'])
        self.assertEqual(set(files), {'Ordinary.json'})
        data[0]['mapping']['entry']['message']['content']['parts'] = ['Updated synthetic message']
        source.write_text(json.dumps(data))
        after = run()
        self.assertEqual(set(after), {'Ordinary.json'})
        self.assertEqual(after['Ordinary.json']['messages'][0]['content'], ['Updated synthetic message'])
