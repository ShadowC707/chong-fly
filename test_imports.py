import sys, os
from importlib import import_module
sys.path.insert(0, os.path.abspath('.'))

for root, dirs, files in os.walk('.'):
    if 'venv' in root or '.git' in root or '.idea' in root:
        continue
    for f in files:
        if f.endswith('.py'):
            mod = os.path.join(root, f).replace('.\\', '').replace('\\', '.').replace('/', '.').replace('.py', '')
            if mod == 'test_imports': continue
            try:
                import_module(mod)
            except Exception as e:
                print(f"Failed {mod}: {e}")
