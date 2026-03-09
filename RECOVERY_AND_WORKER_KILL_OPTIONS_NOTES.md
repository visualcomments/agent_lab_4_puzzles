# Recovery and worker kill options

This patch adds two operational features:

- An automatic g4f recovery stage before offline baseline fallback.
  - Controlled by `AGENTLAB_G4F_RECOVERY_ROUNDS` (default 1),
    `AGENTLAB_G4F_RECOVERY_MAX_ITERS` (default 2), and
    `AGENTLAB_G4F_RECOVERY_SLEEP_S` (default 1.5).
  - Exposed in `pipeline_cli.py` and `run_perm_pipeline.py` via
    `--g4f-recovery-rounds`, `--g4f-recovery-max-iters`, and
    `--g4f-recovery-sleep`.
- Optional disabling of hard process-group termination for timed-out workers.
  - Controlled by `AGENTLAB_WORKER_KILL_PROCESS_GROUP=0`.
  - Exposed via `--worker-no-kill-process-group`.

The recovery stage is triggered only for recoverable remote-model failures such as
format problems (`did not return a python file`) or worker/provider timeouts.
