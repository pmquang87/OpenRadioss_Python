with open('pyradioss/input/prop_reader.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('title, cards, fixed = _data_cards(block)', 'title, cards, fixed = _data_cards(block)\n    print(\"PARSE_SPR_PRE CARDS LEN:\", len(cards))')

with open('pyradioss/input/prop_reader.py', 'w', encoding='utf-8') as f:
    f.write(text)
