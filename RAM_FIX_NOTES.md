# RSS guard only

This version keeps only the RSS-based memory guard.

Included:
- RSS guard in `AgentLaboratory/perm_pipeline/run_perm_pipeline.py`
- uses `psutil.Process().memory_info().rss`
- configurable via `AGENTLAB_MAX_RSS_MB`
- auto-enables in Colab-like environments at about 72% of total RAM

Not included:
- no automatic remote `--max-iters` cap
- no `AGENTLAB_REMOTE_MAX_ITERS_CAP`
- no `AGENTLAB_ALLOW_HUGE_MAX_ITERS`
