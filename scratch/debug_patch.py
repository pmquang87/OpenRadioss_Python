with open('pyradioss/input/prop_reader.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('params[\"ilock\"] = _iv(h[3])', 'params[\"ilock\"] = _iv(h[3])\n        print(\"SETTING SENS_ID!\", params[\"sens_id\"])')

with open('pyradioss/input/prop_reader.py', 'w', encoding='utf-8') as f:
    f.write(text)
