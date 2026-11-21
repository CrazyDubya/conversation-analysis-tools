"""Consumer regressions for short labels, phrase matching and optional timestamps."""
import sqlite3
import pandas as pd
import pytest
from conversation_analysis.pipeline.duplicate_detector import DuplicateDetector
from conversation_analysis.pipeline.relevance_scorer import RelevanceScorer
from conversation_analysis.core.search_engine import SearchEngine


def test_short_labels_remain_distinct_across_duplicate_consumers():
    detector = DuplicateDetector(similarity_threshold=0.9)
    documents = [(10, 'This is document A.'), (11, 'This is document A.'),
                 (20, 'This is document B.'), (21, 'This is document B.'),
                 (30, 'This is document C.')]
    assert {frozenset(x) for x in detector.cluster_duplicates(documents)} == {frozenset({10, 11}), frozenset({20, 21})}
    assert detector.get_unique_documents(documents) == [10, 20, 30]
    near = detector.find_near_duplicates(documents, 20)
    assert [doc_id for doc_id, _ in near] == [21]
    assert near[0][1] == pytest.approx(1.0)
    matrix, ids = detector.build_similarity_matrix(documents)
    assert ids == [10, 11, 20, 21, 30]
    assert matrix[0, 2] < 0.9
    assert matrix[2, 3] == pytest.approx(1)
    assert detector.text_similarity('AI systems', 'ML systems') < 0.9
    assert detector.text_similarity('AI systems', 'ai systems') == pytest.approx(1)
    assert 'is' not in detector.tokenize('This is an AI system')


@pytest.mark.parametrize('text,expected_density,expected_coverage', [
    ('Machine learning and deep learning.', 1.0, 1.0),
    ('MACHINE learning!', 1.0, 0.5),
    ('Machine and learning.', 0.0, 0.0),
    ('Machine learningish.', 0.0, 0.0),
    ('Unrelated document.', 0.0, 0.0),
    ('', 0.0, 0.0),
    (None, 0.0, 0.0),
])
def test_phrase_scores_require_contiguous_whole_words(text, expected_density, expected_coverage):
    scorer = RelevanceScorer(['machine learning', 'deep learning'])
    assert scorer.keyword_density_score(text) == pytest.approx(expected_density)
    assert scorer.keyword_coverage_score(text) == pytest.approx(expected_coverage)


def test_density_counts_occurrences_without_overlap_inflation():
    scorer = RelevanceScorer(['machine learning', 'learning'])
    assert scorer.keyword_density_score('machine learning learning document') == pytest.approx(3 / 4)
    assert scorer.keyword_coverage_score('machine learning learning document') == 1
    single = RelevanceScorer(['learning'])
    assert single.keyword_density_score('learning learning document') == pytest.approx(2 / 3)
    assert single.keyword_coverage_score('learning document') == 1
    assert RelevanceScorer([]).keyword_density_score('learning') == 0


@pytest.mark.parametrize('has_updated_at', [False, True])
def test_date_search_preserves_optional_timestamp_without_migration(tmp_path, has_updated_at):
    path = tmp_path / 'synthetic.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE conversations(id TEXT PRIMARY KEY, title TEXT, platform TEXT, created_at TEXT' +
                           (', updated_at TEXT' if has_updated_at else '') + ')')
        connection.execute('CREATE TABLE messages(id TEXT, conversation_id TEXT)')
        values = ['a', 'Fixture A', 'claude', '2024-01-01'] + (['2024-01-03'] if has_updated_at else [])
        connection.execute('INSERT INTO conversations VALUES (' + ','.join('?' for _ in values) + ')', values)
        values = ['b', 'Fixture B', 'chatgpt', '2024-01-02'] + ([None] if has_updated_at else [])
        connection.execute('INSERT INTO conversations VALUES (' + ','.join('?' for _ in values) + ')', values)
        connection.executemany('INSERT INTO messages VALUES (?, ?)', [('m1', 'a'), ('m2', 'a')])
    before = path.read_bytes()
    with SearchEngine(str(path)) as search:
        result = search.date_range_search('2024-01-01', '2024-01-02')
        assert set(result.conversation_id) == {'a', 'b'}
        record = result.set_index('conversation_id')
        assert record.loc['a', 'message_count'] == 2
        assert record.loc['b', 'message_count'] == 0
        if has_updated_at:
            assert record.loc['a', 'updated_at'] == '2024-01-03'
        else:
            assert pd.isna(record.loc['a', 'updated_at'])
        assert pd.isna(record.loc['b', 'updated_at'])
        filtered = search.date_range_search('2024-01-01', '2024-01-02', platform='claude')
        assert list(filtered.conversation_id) == ['a']
        assert len(search.date_range_search('2024-01-02', '2024-01-02')) == 1
        assert search.date_range_search('2030-01-01', '2030-01-02').empty
        assert len(search.date_range_search('2024-01-01', '2024-01-02', limit=1)) == 1
    assert path.read_bytes() == before
