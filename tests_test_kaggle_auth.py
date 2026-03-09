from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'llm-puzzles'))

from src import kaggle_utils  # type: ignore


def test_build_kaggle_env_supports_legacy_json(tmp_path):
    creds = tmp_path / 'kaggle.json'
    creds.write_text('{"username": "demo-user", "key": "demo-key"}', encoding='utf-8')

    env = kaggle_utils.build_kaggle_env(str(creds), config_dir=str(tmp_path / 'cfg'))

    assert env['KAGGLE_USERNAME'] == 'demo-user'
    assert env['KAGGLE_KEY'] == 'demo-key'
    cfg_dir = Path(env['KAGGLE_CONFIG_DIR'])
    assert (cfg_dir / 'kaggle.json').exists()
    assert not (cfg_dir / 'access_token').exists()



def test_build_kaggle_env_supports_plain_access_token(tmp_path):
    token_file = tmp_path / 'access_token.txt'
    token_file.write_text('token-123', encoding='utf-8')

    env = kaggle_utils.build_kaggle_env(str(token_file), config_dir=str(tmp_path / 'cfg2'))

    assert env['KAGGLE_API_TOKEN'] == 'token-123'
    cfg_dir = Path(env['KAGGLE_CONFIG_DIR'])
    assert (cfg_dir / 'access_token').read_text(encoding='utf-8') == 'token-123'
    assert 'KAGGLE_USERNAME' not in env
    assert 'KAGGLE_KEY' not in env
