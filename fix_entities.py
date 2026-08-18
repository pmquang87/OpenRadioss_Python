"""
Find all entity constructor calls in starter_keywords.py that use fields
not present in the actual entity class definitions.
"""
import re
import ast
import sys

# Load the entities module to get actual class fields
sys.path.insert(0, '.')
from pyradioss.model import entities

# Get all dataclass fields for each entity
entity_fields = {}
for name in dir(entities):
    obj = getattr(entities, name)
    if isinstance(obj, type) and hasattr(obj, '__dataclass_fields__'):
        entity_fields[name] = set(obj.__dataclass_fields__.keys())

# Now scan starter_keywords.py for constructor calls
with open('pyradioss/input/starter_keywords.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find patterns like EntityClass(field1=..., field2=...)
# Simple regex approach - find "ClassName(" then extract keyword args
pattern = re.compile(r'(\w+)\(\s*\n?\s*(?:[\w.]+=)', re.MULTILINE)

issues = []
for m in pattern.finditer(content):
    class_name = m.group(1)
    if class_name not in entity_fields:
        continue
    
    # Get the full constructor call
    start = m.start()
    # Find matching paren
    depth = 0
    pos = content.index('(', start)
    end = pos
    for i in range(pos, min(pos + 2000, len(content))):
        if content[i] == '(':
            depth += 1
        elif content[i] == ')':
            depth -= 1
            if depth == 0:
                end = i
                break
    
    call_text = content[start:end+1]
    
    # Extract keyword argument names
    kwarg_pattern = re.compile(r'(\w+)\s*=')
    kwargs = kwarg_pattern.findall(call_text)
    
    valid_fields = entity_fields[class_name]
    bad_fields = [k for k in kwargs if k not in valid_fields and not k.startswith('_')]
    
    if bad_fields:
        line_no = content[:start].count('\n') + 1
        issues.append((class_name, line_no, bad_fields))

if issues:
    print(f"Found {len(issues)} constructor calls with mismatched fields:")
    for cls, line, fields in issues:
        print(f"  Line {line}: {cls}({', '.join(fields)}=...) - fields not in class: {fields}")
        print(f"    Valid fields: {sorted(entity_fields[cls])}")
else:
    print("No mismatched constructor calls found!")
