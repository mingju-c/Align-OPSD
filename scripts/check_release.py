#!/usr/bin/env python3
"""Check source syntax and submission hygiene without importing ML dependencies."""
import ast
from pathlib import Path
import re
import subprocess
import warnings

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = { '.pytest_cache', '.ruff_cache', '__pycache__', 'logs', 'checkpoints', 'outputs', 'wandb', 'reports'}
# Match literal machine-specific roots, not project-relative data/ paths.
MACHINE_PATH = re.compile(r'(?<![\w${}/])/(?:root|home|mnt|workspace|scratch|Users|opt/tiger)/')
TEXT = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.md', '.txt', '.rst', '.slurm'}


def main():
    failures = []
    counts = {'python': 0, 'shell': 0}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        for path in sorted(ROOT.rglob('*')):
            rel = path.relative_to(ROOT)
            # Git metadata is expected in a checkout; GitHub workflows are source.
            if '.git' in rel.parts:
                continue
            if any(part in FORBIDDEN for part in rel.parts):
                if path.is_dir() and path.name in FORBIDDEN:
                    failures.append(f'Runtime/private directory: {rel}')
                continue
            if path.is_symlink():
                failures.append(f'Symlink must be resolved before submission: {rel}')
                continue
            if not path.is_file():
                continue
            if path.suffix in TEXT:
                text = path.read_text(encoding='utf-8')
                for number, line in enumerate(text.splitlines(), 1):
                    if MACHINE_PATH.search(line):
                        failures.append(f'Machine-specific path: {rel}:{number}')
            if path.suffix == '.py':
                counts['python'] += 1
                try:
                    ast.parse(path.read_text(), filename=str(rel))
                except SyntaxError as exc:
                    failures.append(str(exc))
            elif path.suffix == '.sh':
                counts['shell'] += 1
                result = subprocess.run(['bash', '-n', str(path)], text=True, capture_output=True)
                if result.returncode:
                    failures.append(result.stderr.strip())
    print(f'Checked {counts["python"]} Python files and {counts["shell"]} shell scripts.')
    if failures:
        print('\n'.join(failures))
        return 1
    print('PASS: syntax, machine-specific paths, symlinks and excluded directories.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
