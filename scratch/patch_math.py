with open('pyradioss/input/prop_reader.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.startswith('from typing import'):
        lines.insert(i, 'import math\n')
        break
with open('pyradioss/input/prop_reader.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
