import numpy as np
import pytest

from varhadi_data.align import clean_for_alignment, ctc_align, group_units
from varhadi_data.vahnt_text import parse_chapter_html

BLANK = 5
V = 6


def synthetic_log_probs(frames: list[int]) -> np.ndarray:
    """One frame per entry: that class gets prob 0.9, the rest share 0.1."""
    lp = np.full((len(frames), V), np.log(0.1 / (V - 1)))
    lp[np.arange(len(frames)), frames] = np.log(0.9)
    return lp


def test_ctc_align_finds_tokens_and_skips_unscripted_intro():
    # frames 0-3: speech that is NOT in the text (tokens 3,4 = an unscripted announcement)
    frames = [3, 4, 3, 4, BLANK, 0, 0, BLANK, 1, 1, 1, BLANK, 2, BLANK, BLANK]
    spans = ctc_align(synthetic_log_probs(frames), [0, 1, 2], BLANK)
    assert spans == [(5, 6), (8, 10), (12, 12)]


def test_ctc_align_handles_repeated_tokens_and_trailing_audio():
    frames = [BLANK, 1, BLANK, 1, BLANK, 3, 3, 4]  # 3,4 at the end: unscripted outro
    spans = ctc_align(synthetic_log_probs(frames), [1, 1], BLANK)
    assert spans == [(1, 1), (3, 3)]


def test_ctc_align_long_sequence_beyond_int8_states():
    tokens = [i % 5 for i in range(200)]  # 401 states; no adjacent repeats
    frames = [BLANK, BLANK] + [tok for tok in tokens for _ in (0, 1)] + [BLANK]
    spans = ctc_align(synthetic_log_probs(frames), tokens, BLANK)
    assert spans[0] == (2, 3) and spans[-1] == (400, 401)


def test_ctc_align_rejects_impossible_input():
    with pytest.raises(ValueError, match="too short"):
        ctc_align(synthetic_log_probs([1, 1]), [1, 1], BLANK)  # repeat needs a blank between
    with pytest.raises(ValueError, match="no tokens"):
        ctc_align(synthetic_log_probs([1]), [], BLANK)


def test_group_units_respects_max_and_cuts_in_gaps():
    spans = [(1.0, 6.0), (7.0, 12.0), (13.0, 30.0), (30.4, 50.0), (51.0, 55.0)]
    groups = group_units(spans, max_sec=12.0)
    assert [(a, b) for a, b, _, _ in groups] == [(0, 1), (2, 2), (3, 3), (4, 4)]
    assert groups[0][2] == 1.0 and groups[0][3] == 12.5  # midpoint of the 12-13 gap
    assert groups[1][2] == 12.5  # contiguous: next segment starts at the same cut
    assert groups[-1][3] == 55.0


def test_clean_for_alignment_keeps_matras_and_drops_punctuation():
    text = "येशूनं त्याले म्हतलं, “मी तुले पायलं.” मोठे-मोठे।"
    assert clean_for_alignment(text) == "येशूनं त्याले म्हतलं मी तुले पायलं मोठे मोठे"


PAGE = """
<div class="X__chapter" data-usfm="JHN.1"><div class="X__label">1</div>
<div class="X__s"><span class="X__heading">प्रारंभी शब्द होता</span></div>
<div class="X__p"><span class="X__verse" data-usfm="JHN.1.1"><span class="X__label">1</span>
<span class="X__content">पयले शब्द होता, </span><span class="X__note X__f"><span class="X__body">टीप</span></span>
<span class="X__content">अन् हा शब्द देवासोबत होता.</span></span></div>
<div class="X__r"><span class="X__heading">(</span>मत्तय 3:1-12<span class="X__heading">)</span></div>
<div class="X__p"><span class="X__verse" data-usfm="JHN.1.2"><span class="X__content">अन् तोच<br>देव होता.</span></span></div>
<div class="X__p"><span class="X__verse" data-usfm="JHN.1.2"><span class="X__content">दुसरा भाग.</span></span></div>
</div>"""


def test_parse_chapter_keeps_read_text_only():
    assert parse_chapter_html(PAGE) == [
        {"kind": "heading", "ref": "", "text": "प्रारंभी शब्द होता"},
        {"kind": "verse", "ref": "JHN.1.1", "text": "पयले शब्द होता, अन् हा शब्द देवासोबत होता."},
        {"kind": "verse", "ref": "JHN.1.2", "text": "अन् तोच देव होता. दुसरा भाग."},  # split verse merged
    ]
