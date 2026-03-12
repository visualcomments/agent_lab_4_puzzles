# Megaminx module added

This repository now contains a real offline competition bundle for `cayley-py-megaminx`:
- official `data/puzzle_info.json`
- official `data/test.csv`
- official `data/sample_submission.csv`
- a validator that checks generator legality and whether the path reaches `central_state`
- a baseline solver that replays the official sample path for any known test state
- prompts tailored to the actual Megaminx rules and data format
- submission helpers and regression tests
