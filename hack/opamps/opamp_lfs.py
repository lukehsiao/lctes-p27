from fonduer.utils.data_model_utils import (
    get_col_ngrams,
    get_horz_ngrams,
    get_page,
    get_row_ngrams,
    get_vert_ngrams,
    overlap,
)

ABSTAIN = 0
TRUE = 1
FALSE = 2


def neg_low_page_num(c):
    if get_page(c[0]) > 8:
        return FALSE

    return ABSTAIN


def pos_gain(c):
    row_ngrams = set(get_row_ngrams(c.gain, lower=True))
    if overlap(["gain"], row_ngrams):
        return TRUE
    else:
        ABSTAIN


def pos_gain_keywords(c):
    vert_ngrams = set(get_vert_ngrams(c.gain, n_max=1, lower=True))
    row_ngrams = set(get_row_ngrams(c.gain, lower=True))
    if overlap(["typ", "typ."], vert_ngrams) and overlap(["khz", "mhz"], row_ngrams):
        return TRUE

    return ABSTAIN


def neg_keywords(c):
    horz_ngrams = set(get_horz_ngrams(c.gain, lower=True))
    if overlap(["bandwidth"], horz_ngrams) and not overlap(["gain"], horz_ngrams):
        return FALSE

    return ABSTAIN


def pos_sen_lf(c):
    if (
        pos_gain(c) == TRUE
        and pos_gain_keywords(c) == TRUE
        and neg_keywords(c) == ABSTAIN
    ):
        return TRUE

    return FALSE


# Supply Current LFs
def pos_current(c):
    row_ngrams = list(get_row_ngrams(c.supply_current))
    keywords = ["supply", "quiescent", "iq", "is", "idd"]
    return TRUE if overlap(keywords, row_ngrams) else ABSTAIN


def pos_current_units(c):
    row_ngrams = list(get_row_ngrams(c.supply_current))
    current_units = ["ma", "μa", "ua", "µa", "\uf06da"]
    return TRUE if overlap(current_units, row_ngrams) else ABSTAIN


def pos_current_typ(c):
    return (
        TRUE
        if overlap(["typ", "typ."], get_col_ngrams(c.supply_current, lower=True))
        else ABSTAIN
    )


def neg_current_keywords_in_column(c):
    return (
        FALSE
        if overlap(
            ["over", "temperature", "vgn", "f", "-3", "db", "dbc", "min", "max"],
            get_col_ngrams(c.supply_current, lower=True),
        )
        else ABSTAIN
    )


def neg_current_keywords_in_vert(c):
    return (
        FALSE
        if overlap(
            ["over", "temperature", "vgn", "f", "-3", "db", "dbc", "min", "max"],
            get_vert_ngrams(c.supply_current, lower=True),
        )
        else ABSTAIN
    )


def neg_current_keywords_in_row(c):
    return (
        FALSE
        if overlap(
            ["output", "drive", "voltage", "io"],
            get_row_ngrams(c.supply_current, lower=True),
        )
        else ABSTAIN
    )


gain_lfs = [pos_sen_lf]

current_lfs = [
    neg_current_keywords_in_column,
    neg_current_keywords_in_row,
    neg_current_keywords_in_vert,
    neg_low_page_num,
    pos_current,
    pos_current_typ,
    pos_current_units,
]
