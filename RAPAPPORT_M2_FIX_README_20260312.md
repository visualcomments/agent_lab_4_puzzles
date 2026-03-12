# Rapapport M2 fix

Исправлена перепутанная семантика ходов `I` и `S` для соревнования `cayleypy-rapapport-m2`.

Что изменено:
- `competitions/cayleypy-rapapport-m2/solve_module.py` — выходные токены переименовываются из внутренней исторической семантики в семантику соревнования.
- `competitions/cayleypy-rapapport-m2/validate_solve_output.py` — локальный валидатор приведён к правилам соревнования.
- `tests_test_rapapport_guardrails.py` — добавлен тест на эталонный путь из sample submission.
