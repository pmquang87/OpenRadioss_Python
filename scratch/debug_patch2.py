with open('pyradioss/input/prop_reader.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('def parse_spr_pre(block: KeywordBlock, log: MessageLog) -> Optional[Property]:', 'def parse_spr_pre(block: KeywordBlock, log: MessageLog) -> Optional[Property]:\n    print(\"HULLO FROM PARSE_SPR_PRE!\")')

with open('pyradioss/input/prop_reader.py', 'w', encoding='utf-8') as f:
    f.write(text)
