import json
from pathlib import Path

def test_opencode_config():
 d=json.loads(Path('opencode.json').read_text()); p=d['provider']['deepseek-modal']
 assert p['npm']=='@ai-sdk/openai-compatible'; assert 'deepseek-v4-flash' in p['models']

def test_b200():
 t=Path('modal/app.py').read_text(); assert "GPU_COUNT=int(os.getenv('GPU_COUNT','4'))" in t; assert "gpu=f'{GPU_TYPE}:{GPU_COUNT}'" in t

def test_popen_fix():
 t=Path('modal/app.py').read_text(); assert 'stdout=self.log_file' in t; assert 'stdin=subprocess.DEVNULL' in t; assert 'start_new_session=True' not in t
