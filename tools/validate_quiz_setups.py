"""Execute each sandbox practice-answer setup in an independent Python namespace."""
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def validate():
    pages = json.loads((ROOT / 'EAGE_pythonSandBox/generated/course.json').read_text(encoding='utf-8'))['pages']
    count = 0
    with patch('builtins.input', return_value='7'):
        for page in pages:
            if page['group'] != 'Quizzes':
                continue
            for index, cell in enumerate(page['codes']):
                if not (cell['solution'] and cell['runnable']):
                    continue
                code = '\n\n'.join(c['code'] for c in page['codes'][:index + 1] if c['runnable'])
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        exec(compile(code, page['id'] + ':' + cell['id'], 'exec'), {'__name__': '__main__'})
                except Exception as exc:
                    raise RuntimeError(f"Practice setup failed: {page['id']}:{cell['id']}") from exc
                finally:
                    plt.close('all')
                count += 1
    if count != 97:
        raise ValueError(f'Expected 97 runnable quiz answers; found {count}')
    print(f'Executed {count} independently assembled quiz practice-answer setups.')


if __name__ == '__main__':
    validate()
