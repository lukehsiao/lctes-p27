import logging
import re

from fonduer.utils.data_model_utils import (
    get_horz_ngrams,
    get_row_ngrams,
    get_vert_ngrams,
    is_horz_aligned,
    is_vert_aligned,
    same_table,
)

from hack.transistors.transistor_spaces import expand_part_range

logger = logging.getLogger(__name__)

part_pattern = re.compile(r"^([0-9]+[A-Z]+|[A-Z]+[0-9]+)[0-9A-Z]*$")
polarity_pattern = re.compile(r"NPN|PNP", re.IGNORECASE)


def _filter_non_parts(c):
    ret = set()
    for _ in c:
        for __ in expand_part_range(_):
            if part_pattern.match(__) and len(__) > 2:
                ret.add(__)
    return ret


def stg_temp_filter(c):
    (part, attr) = c
    if same_table((part, attr)):
        return is_horz_aligned((part, attr)) or is_vert_aligned((part, attr))

    # Filter if not inside of a table
    return part.context.sentence.is_tabular()


def polarity_filter(c):
    (part, attr) = c

    # Check if the polarities are not matched with the part
    ngrams_part = set(
        x
        for x in get_row_ngrams(part, n_max=1, lower=False)
        if (x and polarity_pattern.match(x))
    )
    if len(ngrams_part) != 0 and all(
        not attr.context.get_span().lower().startswith(_.lower()) for _ in ngrams_part
    ):
        logger.debug(
            f"ngrams_part: {ngrams_part}\nattr: {attr.context.get_span().lower()}"
        )
        return False

    if same_table(c):
        return is_horz_aligned(c) or is_vert_aligned(c)

    return True


def ce_v_max_filter(c):
    (part, attr) = c
    if same_table(c):
        return is_horz_aligned(c) or is_vert_aligned(c)

    # Check if the ce_v_max's are not matched with the part
    ngrams_part = _filter_non_parts(set(x for x in get_vert_ngrams(attr, n_max=1)))
    ngrams_part = _filter_non_parts(
        ngrams_part.union(set(x for x in get_horz_ngrams(attr, n_max=1)))
    )

    if len(ngrams_part) != 0 and all(
        not part.context.get_span().lower().startswith(_.lower()) for _ in ngrams_part
    ):
        logger.debug(
            f"ngrams_part: {ngrams_part}\npart: {part.context.get_span().lower()}"
        )
        return False

    return True
