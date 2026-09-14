# conversation-analysis-tools

Consolidated toolkit for analyzing exported ChatGPT / Claude conversations. Merged 2026-09-12
from five Stephen Thompson repositories into this one home; the source repos were archived after the
merge. Full file-by-file lineage lives in [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Layout

```
src/conversation_analysis/
├── core/            # search engine, database, config (from the original base repo)
├── pipeline/       # priority classification, dedup, relevance, summarization pipeline
├── ingest/         # archive_parser  (ChatGPT export -> parsed messages)
│                   # html_extract    (HTML `var jsonData` -> JSON, recursive regex)
│                   # aicode          (non-functional API stub, kept for provenance)
├── analysis/       # clustering, anomaly, vector_space, duo_compare, claude_db,
│                   # conversation_stats, analyzegpt, combined_analytics, ...
├── viz/            # wordcloud, heatmaps, graphs (conversation flowcharts)
├── schema_tools/   # LLM-gated SQL schema-change validator (quarantined; candidate for spin-out)
└── config/         # pipeline_config.yaml
apps/macos/         # GPT-Analyze2 macOS app, copied verbatim from CrazyDubya/GPT-Analyze2
tests/              # pytest suite (core/pipeline + ingest)
docs/PROVENANCE.md  # every file mapped to its source repo + commit
```

Root scripts (`analyze_conversations.py`, `run_pipeline.py`, `conversation_search_gui.py`, …) are
the original base-repo entry points and now import from the `conversation_analysis` package.

## Install

```bash
pip install -r requirements.txt
# Optional/heavy extras (Anthropic/OpenAI clients, graphviz, pyvis, python-louvain, sqlparse):
pip install -r requirements-optional.txt
python setup.py install        # or: pip install .
```

## Usage

```bash
analyze-content --help                    # pipeline CLI entry point
python -m conversation_analysis.ingest.html_extract --input page.html --output data.json
python -m conversation_analysis.schema_tools.main original.sql proposed.sql "$LLM_API_KEY"
python src/conversation_analysis/analysis/clustering.py --input conversations.json
```

Input paths are configurable: every ported script that once hardcoded a
`/Users/puppuccino/...` path now takes a CLI argument or a `CONVERSATIONS_JSON` /
`CONV_ANALYSIS_INPUT_DIR` environment variable. See `docs/PROVENANCE.md` for the
twin-file decisions (e.g. which of two near-identical scripts was kept).

## Tests

```bash
pytest
```

## Security note

`src/conversation_analysis/ingest/aicode.py` is a non-functional stub preserved for provenance;
it reads `ANTHROPIC_API_KEY` from the environment and contains no key. The archived source
repos keep their full git history, so **rotate/revoke the historical Anthropic API key** used by
the old `ai-monitoring-core` experiments.

## License

MIT — same as the source repositories.

## Wave 5 consolidation (2026-09-14)

| Source repo | Subdirectory | Merged HEAD |
|---|---|---|
| `CrazyDubya/rabbitmq-llm-chat` (`master`) | `rabbitmq-llm-chat/` | `6fe07582` |

| `CrazyDubya/WikipediaANIReview` (`main`) | `wikipedia-ani-review/` | `4a8249ad` |

Sources were archived after byte-identical verification. File-by-file lineage in `docs/PROVENANCE.md`.
