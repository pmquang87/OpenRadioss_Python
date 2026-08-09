with open('pyradioss/input/prop_reader.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Restore from backup or just fix it directly
import shutil
shutil.copy('pyradioss/input/prop_reader.py', 'scratch/prop_reader_bak.py')

# Clean up debug prints
import re
text = re.sub(r'print\(\"HULLO FROM PARSE_SPR_PRE\!\"\)\n    ', '', text)
text = re.sub(r'\n    print\(\"PARSE_SPR_PRE CARDS LEN:\", len\(cards\)\)', '', text)
text = re.sub(r'head = _get\(cards, 0\)\n    print\(\"HEAD IS:\", head\.raw if head else None\)', 'head = _get(cards, 0)', text)
text = re.sub(r'params\[\"ilock\"\] = _iv\(h\[3\]\)\n        print\(\"SENS_ID IS SET\!\", params\[\"sens_id\"\]\)', 'params[\"ilock\"] = _iv(h[3])', text)

# Add defaults!
text = text.replace('params = _universal_geo_params()\n    head = _get(cards, 0)', 'params = _universal_geo_params()\n    params[\"sens_id\"] = 0\n    params[\"ilock\"] = 0\n    head = _get(cards, 0)')

with open('pyradioss/input/prop_reader.py', 'w', encoding='utf-8') as f:
    f.write(text)
