from pyradioss.input.deck_reader import KeywordBlock, Card
import pyradioss.input.starter_keywords as sk

c2 = Card("      1001               1.000000000E-03       0.01.00000000000000E+301.00000000000000E+30")
c2.raw = "      1001               1.000000000E-03       0.01.00000000000000E+301.00000000000000E+30"

t3 = c2.tokens()
if len(t3) < 4 and len(c2.raw) > 20:
    c = c2.raw.rstrip('\n').ljust(90)
    t3 = [c[0:10], c[10:30], c[30:50], c[50:70], c[70:80], c[80:90]]
    t3 = [x.strip() for x in t3 if x.strip()]

print(t3)
