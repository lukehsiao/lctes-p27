import logging
import re

from fonduer.candidates import MentionNgrams
from fonduer.candidates.models.implicit_span_mention import TemporaryImplicitSpanMention

logger = logging.getLogger(__name__)


class MentionNgramsCurrent(MentionNgrams):
    def __init__(self, n_max=2, split_tokens=["-", "/"]):
        super(MentionNgrams, self).__init__(n_max=n_max, split_tokens=split_tokens)

    def apply(self, doc):
        for ts in MentionNgrams.apply(self, doc):
            m = re.match(r"^(±)?\s*(\d+)\s*(\.)?\s*(\d*)$", ts.get_span())
            if m:
                # Handle case that random spaces are inserted (e.g. "± 2  . 3")
                temp = ""
                if m.group(1):
                    temp += m.group(1)
                if m.group(2):
                    temp += m.group(2)
                if m.group(3):
                    temp += m.group(3)
                if m.group(4):
                    temp += m.group(4)

                yield TemporaryImplicitSpanMention(
                    sentence=ts.sentence,
                    char_start=ts.char_start,
                    char_end=ts.char_end,
                    expander_key="opamp_exp",
                    position=0,
                    text=temp,
                    words=[temp],
                    lemmas=[temp],
                    pos_tags=[ts.get_attrib_tokens("pos_tags")[-1]],
                    ner_tags=[ts.get_attrib_tokens("ner_tags")[-1]],
                    dep_parents=[ts.get_attrib_tokens("dep_parents")[-1]],
                    dep_labels=[ts.get_attrib_tokens("dep_labels")[-1]],
                    page=[ts.get_attrib_tokens("page")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    top=[ts.get_attrib_tokens("top")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    left=[ts.get_attrib_tokens("left")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    bottom=[ts.get_attrib_tokens("bottom")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    right=[ts.get_attrib_tokens("right")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    meta=None,
                )
            else:
                yield ts
