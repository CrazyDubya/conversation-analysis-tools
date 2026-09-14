# Provenance

File-by-file lineage for the 2026-09-12 consolidation into
`CrazyDubya/conversation-analysis-tools`. Commit `v2.0-consolidated` (tag) covers the final state.

Source SHAs at merge time:
- `CrazyDubya/conversation-analysis-tools` — `main` (base; restructured, not moved)
- `CrazyDubya/chatgpt-archive-clean` — `f9b2b636` (`master`)
- `CrazyDubya/ai-conversation-analyzer` — `a616deaf` (`master`)
- `CrazyDubya/ai-monitoring-core` — `703acc93` (`master`)
- `CrazyDubya/GPT-Analyze2` — `bcdd20c8` (`main`)
- `CrazyDubya/GPT-Analyze` — intentionally excluded (v1 superseded by GPT-Analyze2)

## Moved files (42)

| New path | Source repo | Source path | Source commit |
|---|---|---|---|
| `src/conversation_analysis/analysis/analyze_haiku.py` | chatgpt-archive-clean | `chatgptarchive-og/analyze-haiku.py` | `f9b2b636` |
| `src/conversation_analysis/analysis/analyze_user_assistant.py` | chatgpt-archive-clean | `chatgptarchive-og/analyzeuserandasst.py` | `f9b2b636` |
| `src/conversation_analysis/analysis/analyzegpt.py` | ai-conversation-analyzer | `analyzegpt2.py` | `a616deaf` |
| `src/conversation_analysis/analysis/anomaly.py` | ai-conversation-analyzer | `anomaly.py` | `a616deaf` |
| `src/conversation_analysis/analysis/claude_db.py` | ai-conversation-analyzer | `claude_db_advanced-2.py` | `a616deaf` |
| `src/conversation_analysis/analysis/clustering.py` | ai-conversation-analyzer | `cluster.py` | `a616deaf` |
| `src/conversation_analysis/analysis/combined_analytics.py` | chatgpt-archive-clean | `chatgptarchive-og/combined_analytics.py` | `f9b2b636` |
| `src/conversation_analysis/analysis/conversation_stats.py` | ai-conversation-analyzer | `conversation_analysis.py` | `a616deaf` |
| `src/conversation_analysis/analysis/convert_conv_analysis.py` | chatgpt-archive-clean | `chatgptarchive-og/convert_conv_analysis.py` | `f9b2b636` |
| `src/conversation_analysis/analysis/duo_compare.py` | ai-conversation-analyzer | `analyzed_conversations-duo-concu.py` | `a616deaf` |
| `src/conversation_analysis/analysis/vector_space.py` | ai-conversation-analyzer | `vector.py` | `a616deaf` |
| `src/conversation_analysis/ingest/aicode.py` | ai-monitoring-core | `aicode.py` | `703acc93` |
| `src/conversation_analysis/ingest/archive_parser.py` | ai-monitoring-core | `chatgptarchive.py` | `703acc93` |
| `src/conversation_analysis/ingest/html_extract.py` | chatgpt-archive-clean | `chatgptarchive/data_extraction.py` | `f9b2b636` |
| `src/conversation_analysis/schema_tools/change_validator.py` | chatgpt-archive-clean | `schema/change_validator.py` | `f9b2b636` |
| `src/conversation_analysis/schema_tools/llm_analyzer.py` | chatgpt-archive-clean | `schema/llm_analyzer.py` | `f9b2b636` |
| `src/conversation_analysis/schema_tools/main.py` | chatgpt-archive-clean | `schema/main.py` | `f9b2b636` |
| `src/conversation_analysis/schema_tools/original_schema.sql` | chatgpt-archive-clean | `schema/original_schema.sql` | `f9b2b636` |
| `src/conversation_analysis/schema_tools/proposed_schema.sql` | chatgpt-archive-clean | `schema/proposed_schema.sql` | `f9b2b636` |
| `src/conversation_analysis/schema_tools/safeguard.py` | chatgpt-archive-clean | `schema/safeguard.py` | `f9b2b636` |
| `src/conversation_analysis/schema_tools/schema_manager.py` | chatgpt-archive-clean | `schema/schema_manager.py` | `f9b2b636` |
| `src/conversation_analysis/viz/graphs.py` | ai-conversation-analyzer | `graph.py` | `a616deaf` |
| `src/conversation_analysis/viz/heatmaps.py` | ai-conversation-analyzer | `heater3.py` | `a616deaf` |
| `src/conversation_analysis/viz/wordcloud.py` | ai-conversation-analyzer | `gptwordcloud-2.py` | `a616deaf` |

| `apps/macos/.gitignore` | GPT-Analyze2 | `.gitignore` | `bcdd20c8` |
| `apps/macos/EFFICIENCY_REPORT.md` | GPT-Analyze2 | `EFFICIENCY_REPORT.md` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2.xcodeproj/project.pbxproj` | GPT-Analyze2 | `GPT_Analyze2.xcodeproj/project.pbxproj` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2.xcodeproj/project.xcworkspace/contents.xcworkspacedata` | GPT-Analyze2 | `GPT_Analyze2.xcodeproj/project.xcworkspace/contents.xcworkspacedata` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2.xcodeproj/project.xcworkspace/xcshareddata/IDEWorkspaceChecks.plist` | GPT-Analyze2 | `GPT_Analyze2.xcodeproj/project.xcworkspace/xcshareddata/IDEWorkspaceChecks.plist` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2.xcodeproj/xcuserdata/puppuccino.xcuserdatad/xcschemes/xcschememanagement.plist` | GPT-Analyze2 | `GPT_Analyze2.xcodeproj/xcuserdata/puppuccino.xcuserdatad/xcschemes/xcschememanagement.plist` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2/Assets.xcassets/AccentColor.colorset/Contents.json` | GPT-Analyze2 | `GPT_Analyze2/Assets.xcassets/AccentColor.colorset/Contents.json` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2/Assets.xcassets/AppIcon.appiconset/Contents.json` | GPT-Analyze2 | `GPT_Analyze2/Assets.xcassets/AppIcon.appiconset/Contents.json` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2/Assets.xcassets/Contents.json` | GPT-Analyze2 | `GPT_Analyze2/Assets.xcassets/Contents.json` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2/ContentView.swift` | GPT-Analyze2 | `GPT_Analyze2/ContentView.swift` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2/GPT_Analyze2.entitlements` | GPT-Analyze2 | `GPT_Analyze2/GPT_Analyze2.entitlements` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2/GPT_Analyze2App.swift` | GPT-Analyze2 | `GPT_Analyze2/GPT_Analyze2App.swift` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2/Preview Content/Preview Assets.xcassets/Contents.json` | GPT-Analyze2 | `GPT_Analyze2/Preview Content/Preview Assets.xcassets/Contents.json` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2Tests/GPT_Analyze2Tests.swift` | GPT-Analyze2 | `GPT_Analyze2Tests/GPT_Analyze2Tests.swift` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2UITests/GPT_Analyze2UITests.swift` | GPT-Analyze2 | `GPT_Analyze2UITests/GPT_Analyze2UITests.swift` | `bcdd20c8` |
| `apps/macos/GPT_Analyze2UITests/GPT_Analyze2UITestsLaunchTests.swift` | GPT-Analyze2 | `GPT_Analyze2UITests/GPT_Analyze2UITestsLaunchTests.swift` | `bcdd20c8` |
| `apps/macos/README.md` | GPT-Analyze2 | `README.md` | `bcdd20c8` |
| `apps/macos/SWARM_ANALYSIS_REPORT.md` | GPT-Analyze2 | `SWARM_ANALYSIS_REPORT.md` | `bcdd20c8` |

## Twin-file decisions (six)

Near-identical scripts existed in both `chatgpt-archive-clean/chatgptarchive-og/` and
`ai-conversation-analyzer/`. The winner is what shipped above; rationale:

1. `analyze-haiku.py` — **identical** in both; kept the `chatgptarchive-og` copy.
2. `convert_conv_analysis.py` — **identical** in both; kept the `chatgptarchive-og` copy.
3. `analyzeuserandasst.py` — kept the **`chatgptarchive-og`** copy: identical logic, but og uses a
   relative `conversations.json` while the ACA copy hardcodes `/Users/puppuccino/...`.
4. `conversation_analysis.py` -> `conversation_stats.py` — kept the **ACA** copy: identical
   extraction logic, but ACA summarizes to 3 sentences (og: 5); hardcoded path replaced with
   `CONVERSATIONS_JSON` env / relative default.
5. `combined_analytics.py` — kept the **`chatgptarchive-og`** copy: identical logic, og uses
   relative paths for input/output.
6. `analyzegpt.py` — kept the **ACA `analyzegpt2.py`**: it contains everything in og `analyzegpt.py`
   plus TF-IDF keyword extraction and word-diversity metrics; hardcoded path replaced with
   `CONVERSATIONS_JSON` env / relative default.

Newer-of-pair (non-identical, clear winner): `analyzed_conversations-duo-concu.py` (threaded +
tqdm) over the non-concurrent `analyzed_conversations-duo.py`; `claude_db_advanced-2.py` (enhanced
heatmap, excludes `__init__`) over `claude_db_advanced.py`; `gptwordcloud-2.py` (stop-word
filtering, sentiment) over `gptwordcloud.py`; `heater3.py` (adds TextBlob sentiment) over `heater.py`.

## Deliberately excluded

- `chatgpt-archive-clean/writer/` — belongs to the writing cluster, not conversation analysis.
- `chatgpt-archive-clean/autom/galactic_overlord/`, `hub/hive.py` — unrelated experiments.
- `ai-monitoring-core/node-map.py` — broken/fake file, no real functionality.
- `GPT-Analyze` (v1) — superseded by GPT-Analyze2, which ships verbatim under `apps/macos/`.
- One-off ACA scripts (`scattergame.py`, `countfiles.py`, `length.py`, `grid.py`, `file.py`,
  `sample_json.py` variants, etc.) — ad-hoc utilities, not library code.
- Large generated text artifacts (`*.txt`/`*.csv` dumps, `*.png` outputs) from both archives.
- `base/deprecated/` — already marked deprecated in the base repo; dropped in the restructure.

## Transformations applied to ported files

- Hardcoded `/Users/puppuccino/...` paths became CLI args and/or environment variables
  (`CONVERSATIONS_JSON`, `CONV_ANALYSIS_INPUT_DIR`, `ANALYZED_DIR`).
- Module-level execution blocks wrapped in `main()` + `argparse` where the script was importable.
- Heavy optional deps (anthropic, python-louvain, pyvis) made lazy/guarded in `claude_db.py`;
  see `requirements-optional.txt`.
- `schema/` flat imports fixed to package-relative; quarantined under `schema_tools/`.
- `aicode.py`: verified via full history search that no hardcoded key exists at any commit; file
  already reads `ANTHROPIC_API_KEY` from the environment. Kept as a non-functional stub.
- Base repo restructured: `core/`, `pipeline/`, `config/` moved under `src/conversation_analysis/`;
  all imports, `setup.py`, CI coverage/lint paths, and the default config path updated.

## Base-repo files (restructured, not moved from elsewhere)

All files under `src/conversation_analysis/core/`, `pipeline/`, `config/`, the root scripts,
`tests/` (except `tests/test_ingest.py`, new), `.github/workflows/ci.yml`, `setup.py`,
`requirements.txt`, and the root docs originate from `conversation-analysis-tools` `main`
itself and were mechanically relocated/updated as described above.

## Security

Archived source repos retain full git history. The historical Anthropic API key used by the
`ai-monitoring-core` experiments must be rotated/revoked by Stephen — this cannot be done by
the assistant and is recorded here as a required follow-up.

## Wave 5 (2026-09-14)

Source SHAs at merge time:
- `CrazyDubya/rabbitmq-llm-chat` — `6fe07582` (`master`) → merged verbatim under `rabbitmq-llm-chat/`


### Moved files — rabbitmq-llm-chat (30)

| New path | Source repo | Source path | Source commit |
|---|---|---|---|
| `rabbitmq-llm-chat/.env.example` | rabbitmq-llm-chat | `.env.example` | `6fe07582` |
| `rabbitmq-llm-chat/.gitignore` | rabbitmq-llm-chat | `.gitignore` | `6fe07582` |
| `rabbitmq-llm-chat/CODE_REVIEW_SUMMARY.md` | rabbitmq-llm-chat | `CODE_REVIEW_SUMMARY.md` | `6fe07582` |
| `rabbitmq-llm-chat/Dockerfile` | rabbitmq-llm-chat | `Dockerfile` | `6fe07582` |
| `rabbitmq-llm-chat/Makefile` | rabbitmq-llm-chat | `Makefile` | `6fe07582` |
| `rabbitmq-llm-chat/README.md` | rabbitmq-llm-chat | `README.md` | `6fe07582` |
| `rabbitmq-llm-chat/__pycache__/chatroom_manager.cpython-312.pyc` | rabbitmq-llm-chat | `__pycache__/chatroom_manager.cpython-312.pyc` | `6fe07582` |
| `rabbitmq-llm-chat/__pycache__/consume.cpython-312.pyc` | rabbitmq-llm-chat | `__pycache__/consume.cpython-312.pyc` | `6fe07582` |
| `rabbitmq-llm-chat/__pycache__/llm_registry.cpython-312.pyc` | rabbitmq-llm-chat | `__pycache__/llm_registry.cpython-312.pyc` | `6fe07582` |
| `rabbitmq-llm-chat/__pycache__/models.cpython-312.pyc` | rabbitmq-llm-chat | `__pycache__/models.cpython-312.pyc` | `6fe07582` |
| `rabbitmq-llm-chat/__pycache__/rabbitmq_module.cpython-312.pyc` | rabbitmq-llm-chat | `__pycache__/rabbitmq_module.cpython-312.pyc` | `6fe07582` |
| `rabbitmq-llm-chat/__pycache__/web_interface.cpython-312.pyc` | rabbitmq-llm-chat | `__pycache__/web_interface.cpython-312.pyc` | `6fe07582` |
| `rabbitmq-llm-chat/chat-start.py` | rabbitmq-llm-chat | `chat-start.py` | `6fe07582` |
| `rabbitmq-llm-chat/chatroom_manager.py` | rabbitmq-llm-chat | `chatroom_manager.py` | `6fe07582` |
| `rabbitmq-llm-chat/config.py` | rabbitmq-llm-chat | `config.py` | `6fe07582` |
| `rabbitmq-llm-chat/consume.py` | rabbitmq-llm-chat | `consume.py` | `6fe07582` |
| `rabbitmq-llm-chat/docker-compose.yml` | rabbitmq-llm-chat | `docker-compose.yml` | `6fe07582` |
| `rabbitmq-llm-chat/index.html` | rabbitmq-llm-chat | `index.html` | `6fe07582` |
| `rabbitmq-llm-chat/index2.html` | rabbitmq-llm-chat | `index2.html` | `6fe07582` |
| `rabbitmq-llm-chat/llm_registry.py` | rabbitmq-llm-chat | `llm_registry.py` | `6fe07582` |
| `rabbitmq-llm-chat/logLLMchats.py` | rabbitmq-llm-chat | `logLLMchats.py` | `6fe07582` |
| `rabbitmq-llm-chat/logger.py` | rabbitmq-llm-chat | `logger.py` | `6fe07582` |
| `rabbitmq-llm-chat/main.py` | rabbitmq-llm-chat | `main.py` | `6fe07582` |
| `rabbitmq-llm-chat/models.py` | rabbitmq-llm-chat | `models.py` | `6fe07582` |
| `rabbitmq-llm-chat/rabbit.py` | rabbitmq-llm-chat | `rabbit.py` | `6fe07582` |
| `rabbitmq-llm-chat/rabbitmq_module.py` | rabbitmq-llm-chat | `rabbitmq_module.py` | `6fe07582` |
| `rabbitmq-llm-chat/requirements.txt` | rabbitmq-llm-chat | `requirements.txt` | `6fe07582` |
| `rabbitmq-llm-chat/tests/__init__.py` | rabbitmq-llm-chat | `tests/__init__.py` | `6fe07582` |
| `rabbitmq-llm-chat/tests/test_core.py` | rabbitmq-llm-chat | `tests/test_core.py` | `6fe07582` |
| `rabbitmq-llm-chat/web_interface.py` | rabbitmq-llm-chat | `web_interface.py` | `6fe07582` |
- `CrazyDubya/WikipediaANIReview` — `4a8249ad` (`main`) → merged verbatim under `wikipedia-ani-review/`


### Moved files — WikipediaANIReview (31)

| New path | Source repo | Source path | Source commit |
|---|---|---|---|
| `wikipedia-ani-review/.gitignore` | WikipediaANIReview | `.gitignore` | `4a8249ad` |
| `wikipedia-ani-review/ASYNC_REFACTOR.md` | WikipediaANIReview | `ASYNC_REFACTOR.md` | `4a8249ad` |
| `wikipedia-ani-review/COMPLETION.md` | WikipediaANIReview | `COMPLETION.md` | `4a8249ad` |
| `wikipedia-ani-review/FEATURES.md` | WikipediaANIReview | `FEATURES.md` | `4a8249ad` |
| `wikipedia-ani-review/IMPLEMENTATION.md` | WikipediaANIReview | `IMPLEMENTATION.md` | `4a8249ad` |
| `wikipedia-ani-review/QUICKSTART.md` | WikipediaANIReview | `QUICKSTART.md` | `4a8249ad` |
| `wikipedia-ani-review/README.md` | WikipediaANIReview | `README.md` | `4a8249ad` |
| `wikipedia-ani-review/TEST_RESULTS_REVIEW.md` | WikipediaANIReview | `TEST_RESULTS_REVIEW.md` | `4a8249ad` |
| `wikipedia-ani-review/analyzer.py` | WikipediaANIReview | `analyzer.py` | `4a8249ad` |
| `wikipedia-ani-review/analyzer_async.py` | WikipediaANIReview | `analyzer_async.py` | `4a8249ad` |
| `wikipedia-ani-review/ani_review.py` | WikipediaANIReview | `ani_review.py` | `4a8249ad` |
| `wikipedia-ani-review/ani_review_async.py` | WikipediaANIReview | `ani_review_async.py` | `4a8249ad` |
| `wikipedia-ani-review/api_scraper.py` | WikipediaANIReview | `api_scraper.py` | `4a8249ad` |
| `wikipedia-ani-review/api_scraper_async.py` | WikipediaANIReview | `api_scraper_async.py` | `4a8249ad` |
| `wikipedia-ani-review/config.example.env` | WikipediaANIReview | `config.example.env` | `4a8249ad` |
| `wikipedia-ani-review/database.py` | WikipediaANIReview | `database.py` | `4a8249ad` |
| `wikipedia-ani-review/demo_async_performance.py` | WikipediaANIReview | `demo_async_performance.py` | `4a8249ad` |
| `wikipedia-ani-review/generate_screenshots.py` | WikipediaANIReview | `generate_screenshots.py` | `4a8249ad` |
| `wikipedia-ani-review/models.py` | WikipediaANIReview | `models.py` | `4a8249ad` |
| `wikipedia-ani-review/profiling_utils.py` | WikipediaANIReview | `profiling_utils.py` | `4a8249ad` |
| `wikipedia-ani-review/requirements.txt` | WikipediaANIReview | `requirements.txt` | `4a8249ad` |
| `wikipedia-ani-review/scraper.py` | WikipediaANIReview | `scraper.py` | `4a8249ad` |
| `wikipedia-ani-review/screenshots/01_async_tests.png` | WikipediaANIReview | `screenshots/01_async_tests.png` | `4a8249ad` |
| `wikipedia-ani-review/screenshots/02_performance_demo.png` | WikipediaANIReview | `screenshots/02_performance_demo.png` | `4a8249ad` |
| `wikipedia-ani-review/screenshots/03_database_init.png` | WikipediaANIReview | `screenshots/03_database_init.png` | `4a8249ad` |
| `wikipedia-ani-review/screenshots/04_cli_help_main.png` | WikipediaANIReview | `screenshots/04_cli_help_main.png` | `4a8249ad` |
| `wikipedia-ani-review/screenshots/05_cli_help_async.png` | WikipediaANIReview | `screenshots/05_cli_help_async.png` | `4a8249ad` |
| `wikipedia-ani-review/screenshots/06_report_output.png` | WikipediaANIReview | `screenshots/06_report_output.png` | `4a8249ad` |
| `wikipedia-ani-review/screenshots/07_nogil_report.png` | WikipediaANIReview | `screenshots/07_nogil_report.png` | `4a8249ad` |
| `wikipedia-ani-review/test_async.py` | WikipediaANIReview | `test_async.py` | `4a8249ad` |
| `wikipedia-ani-review/vote_parser.py` | WikipediaANIReview | `vote_parser.py` | `4a8249ad` |
