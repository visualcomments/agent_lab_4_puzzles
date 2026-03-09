# Stream timeout and worker RAM fix

## Что исправлено

### 1. Защита от зависания на первом streamed-ответе g4f
В `AgentLaboratory/inference.py` добавлен безопасный сбор streamed-ответа:
- поток читается в фоне через daemon-thread;
- введены ограничения:
  - `AGENTLAB_G4F_STREAM_TIMEOUT_S` — общий дедлайн чтения стрима;
  - `AGENTLAB_G4F_STREAM_IDLE_TIMEOUT_S` — idle-timeout между чанками;
- если провайдер открыл стрим и перестал присылать чанки, пайплайн теперь возвращает частичный ответ или завершает попытку, а не висит бесконечно.

### 2. Убран RAM-overflow из-за `capture_output=True`
Вместо `subprocess.run(..., capture_output=True)` для worker-процессов теперь используется:
- `subprocess.Popen(...)`;
- перенаправление stdout/stderr во временные лог-файлы на диске;
- чтение только tail-части логов при ошибке.

Это убирает накопление stdout/stderr worker-процесса в RAM родительского процесса.

### 3. Жесткое завершение подвисших worker-процессов вместе с дочерними
Для Linux worker запускается в отдельной process group (`start_new_session=True`).
При timeout выполняется kill всей process group.

Это важно для сценариев, где g4f-провайдер поднимает дочерние процессы и после timeout они продолжали держать память.

## Измененные файлы
- `AgentLaboratory/inference.py`
- `AgentLaboratory/perm_pipeline/run_perm_pipeline.py`
- `tests_test_low_ram_optimizations.py`
- `tests_test_codegen_pipeline.py`

## Новые / важные env-переменные
- `AGENTLAB_G4F_STREAM_TIMEOUT_S`
- `AGENTLAB_G4F_STREAM_IDLE_TIMEOUT_S`

По умолчанию код сам выбирает безопасные значения на основе `timeout` запроса.

## Проверка
- `python -m py_compile AgentLaboratory/inference.py AgentLaboratory/perm_pipeline/run_perm_pipeline.py tests_test_low_ram_optimizations.py tests_test_codegen_pipeline.py`
- `pytest -q tests_test_low_ram_optimizations.py tests_test_codegen_pipeline.py`
- `pytest -q tests_test_submission_pipeline.py tests_test_g4f_model_check.py tests_test_local_model_optimizations.py tests_test_agent_model_routing.py tests_test_kaggle_auth.py`
