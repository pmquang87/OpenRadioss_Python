import re
with open('pyradioss/elements/spring.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('print(\"PARAMS:\", prop.params)\n                    st[\"sens_id\"][sl] = prop.params[\"sens_id\"]', 'st[\"sens_id\"][sl] = prop.params.get(\"sens_id\", 0)')
with open('pyradioss/elements/spring.py', 'w', encoding='utf-8') as f:
    f.write(text)
