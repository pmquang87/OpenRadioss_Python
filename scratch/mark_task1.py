with open(r'C:\Users\pmqua\.gemini\antigravity\brain\d65a7818-8e59-4068-b175-e64b77cb6e33\task.md', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('- [ ] Update pyradioss/input/prop_reader.py', '- [x] Update pyradioss/input/prop_reader.py')
text = text.replace('  - [ ] Implement ITYP deduction logic', '  - [x] Implement ITYP deduction logic')
text = text.replace('  - [ ] Read scaling factors', '  - [x] Read scaling factors')
text = text.replace('  - [ ] Change returned object', '  - [x] Change returned object')
text = text.replace('- [ ] Update 	ests/test_m40_residuals.py\n  - [ ] Remove the inactive-property test', '- [x] Update 	ests/test_m40_residuals.py\n  - [x] Remove the inactive-property test')
with open(r'C:\Users\pmqua\.gemini\antigravity\brain\d65a7818-8e59-4068-b175-e64b77cb6e33\task.md', 'w', encoding='utf-8') as f:
    f.write(text)
