import sys
import logging
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import KeywordBlock
from pyradioss.input.prop_reader import parse_spr_pre

lines = [
    "/PROP/SPR_PRE/2",
    "prop_spring",
    "                                                                      ",
    "           13744.468                                                                                ",
    "         1                                                                                          "
]

block = KeywordBlock(lines, "PROP_SPR_PRE_HEAD", False)
log = MessageLog()
prop = parse_spr_pre(block, log)
print(prop.params)
