from pathlib import Path
import py_compile


def test_pipeline_cli_compiles_without_syntax_errors():
    py_compile.compile(str(Path(__file__).resolve().parent / 'pipeline_cli.py'), doraise=True)
