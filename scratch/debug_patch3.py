with open('pyradioss/input/prop_reader.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('params[\"ilock\"] = _iv(h[3])', 'params[\"ilock\"] = _iv(h[3])\n        print(\"SENS_ID IS SET!\", params[\"sens_id\"])')
text = text.replace('head = _get(cards, 0)', 'head = _get(cards, 0)\n    print(\"HEAD IS:\", head.raw if head else None)')

with open('pyradioss/input/prop_reader.py', 'w', encoding='utf-8') as f:
    f.write(text)
