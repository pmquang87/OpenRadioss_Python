from pyradioss.input.deck_reader import KeywordBlock, Card
import pyradioss.input.starter_keywords as sk

c2 = Card("      1001               1.000000000E-03       0.01.00000000000000E+301.00000000000000E+30")
c = c2.raw.rstrip('\n').ljust(100)
# Let's try 10, 20, 20, 20, 10, 10
# Wait! pext is float. If mu is 20 and pext is 20, it would be [10:30] and [30:50].
print('10-20-20-20-10-10:', [c[0:10], c[10:30], c[30:50], c[50:70], c[70:80], c[80:90]])
# Let's try 10-10-20-10-20-20? No.
