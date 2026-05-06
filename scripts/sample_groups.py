from __future__ import annotations


GOOD_ONES: list[str] = [
    "0012-399",  # from the numbered list: 0-6 add them to good ones
    "0024-420",  # from the numbered list: 0-6 add them to good ones
    "0025-260",  # from the numbered list: 0-6 add them to good ones
    "0029+349",  # from the numbered list: 0-6 add them to good ones
    "0112+227",  # from the numbered list: 0-6 add them to good ones
    "0132-169",  # from the numbered list: 0-6 add them to good ones
    "0141+138",  # from the numbered list: 0-6 add them to good ones
    "0205+322",  # Good ones
    "0312-148",  # Good ones
    "0409-179",  # Good ones
    "1522-275",  # Good ones
    "1924+334",  # Good ones
    "2257-364",  # Good ones
    "2341-351",  # Good ones
    "0954+177",  # merged from the numbered follow-up review
    "1156+314",  # merged from the numbered follow-up review
    "1309+119",  # merged from the numbered follow-up review
    "0323+055",  # merged from the numbered follow-up review
    "1056+701",  # merged from the numbered follow-up review
]

NEEDS_BIGGER_IMAGE: list[str] = [
    "1018-317",  # bigger image
    "1354-021",  # source outside field (to the left)
    "2212+018",  # bigger image might help
]

NEEDS_BOX: list[str] = [
    "0653+370",
]

BAD_DATA: list[str] = [
    "0259+077",  # bad streaks / bad data?
    "1246-075",  # ionospheric effects / phase screens (rippling)
    "1719+177",  # L band source offset + RFI (big stripes)
]

NEEDS_MULTITERM: list[str] = [
    "0539-286",
    "0608-223",
    "0653+370",  # frequency structure within source (multiterm nterms>=2 needed in mtmfs deconvolver)
    "2023+544",  # souble source / needs self cal + (freq-dependent cal problem (+- holes effect)mtmf)
    "2137+510",  # souble source / needs self cal + (freq-dependent cal problem (+- holes effect) mtmf)
    "1510-057",  # merged from the numbered follow-up review
    "1513-102",  # merged from the numbered follow-up review
    "1432-180",  # merged from the numbered follow-up review
    "0329+279",  # merged from the numbered follow-up review
    "0005+383",  # merged from the numbered follow-up review
    "0119+321",  # merged from the numbered follow-up review
    "1224+035",  # merged from the numbered follow-up review
]

BAD_ANT: list[str] = [
    "1224+213",  # has amplitude errors (sync with 5 freq band?) or uv-coverage issue
    "1719+177",  # beam size too large + limited uv / bad ant
]

RESOLVED: list[str] = [
    "1246-075",  # uv issue + other → bad baseline? extra source in field? resolved?
    "1949-199",  # double source / needs self cal
    "2023+544",  # souble source / needs self cal + (freq-dependent cal problem (+- holes effect)mtmf)
    "2137+510",  # souble source / needs self cal + (freq-dependent cal problem (+- holes effect) mtmf)
]

BAD_BASELINE: list[str] = [
    "0539-286",  # uv coverage issue (selfcal?)
    "1224+213",  # has amplitude errors (sync with 5 freq band?) or uv-coverage issue
    "1246-075",  # uv issue + other → bad baseline? extra source in field? resolved?
    "1719+177",  # beam size too large + limited uv / bad ant
    "1044+809",  # uv coverage issue
]

BAD_UV_DIST_VS_AMP: list[str] = [
    "0205+322",
    "0025-260",
    "2341-351",
    "1224+213",
    "2212+018",
    "0112+227",
]

EXTRA_SOURCE: list[str] = [
    "1246-075",  # uv issue + other → bad baseline? extra source in field? resolved?
    "1949-199",  # double source / needs self cal
    "2023+544",  # souble source / needs self cal + (freq-dependent cal problem (+- holes effect)mtmf)
    "2137+510",  # souble source / needs self cal + (freq-dependent cal problem (+- holes effect) mtmf)
]

# DONE
BEAM_SIZE_ISSUE: list[str] = [
    "1719+177",  # beam size too large + limited uv / bad ant
]

NEED_SELFCAL: list[str] = [
    "1313+675",  # 40
    "1327+221",  # 41
    "1411+522",  # 46
    "1430+107",  # 49
    "1505+034",  # 54
]

UV_LIM = [
    "0205+322",
    "0259+077",
    "0312-148",
    "1824+107",
    "0042+233",  # merged from the numbered follow-up review
    "0405-131",  # merged from the numbered follow-up review
    "0954+177",  # merged from the numbered follow-up review
    "0956+252",  # merged from the numbered follow-up review
    "1024-008",  # merged from the numbered follow-up review
    "1159+292",  # merged from the numbered follow-up review
    "1327+221",  # merged from the numbered follow-up review
    "1411+522",  # merged from the numbered follow-up review
    "1416+347",  # merged from the numbered follow-up review
    "1927+612",  # merged from the numbered follow-up review
    "2007+404",  # merged from the numbered follow-up review
    "2316+040",  # merged from the numbered follow-up review
    "2330+110",  # merged from the numbered follow-up review
    "2333+390",  # merged from the numbered follow-up review
    "0116-116",  # merged from the numbered follow-up review
    "0416-209",  # merged from the numbered follow-up review
    "0818+423",  # merged from the numbered follow-up review
]
