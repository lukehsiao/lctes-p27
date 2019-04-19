import logging
import pdb

logger = logging.getLogger(__name__)


"""
GENERAL NORMALIZERS
"""


def split_val_condition(input_string):
    """
    Split and return a {'value': v, 'condition': c} dict for the value
    and the condition.
    Condition is empty if no condition was found.

    @param input    A string of the form XXX @ YYYY
    """
    try:
        (value, condition) = [x.strip() for x in input_string.split("@")]
        return {"value": value, "condition": condition}
    except ValueError:
        # no condition was found
        return {"value": input_string.strip(), "condition": None}


def part_family_normalizer(family):
    if family == "Y":
        return str(family)
    elif family == "N":
        return str(family)
    elif family == "N/A":  # Account for Digikey not having any part_family
        return str(family)
    elif family is None:
        return "N/A"
    else:
        logger.error(f"Invalid part family {family}")
        pdb.set_trace()


def doc_normalizer(doc_name):
    if doc_name.endswith(".pdf"):
        return doc_name.split(".pdf")[0]
    elif doc_name.endswith(".PDF"):
        return doc_name.split(".PDF")[0]
    else:
        logger.error(f"Invalid doc_name {doc_name}")
        pdb.set_trace()


def general_normalizer(value):
    # TODO: Right now this is only returning the raw values
    return value.strip()


def manuf_normalizer(manuf):
    return manuf.strip()  # TODO: Make a list of all known manufs


def temperature_normalizer(temperature):
    try:
        (temp, unit) = temperature.rsplit(" ", 1)
        if unit != "C":
            logger.error(f"Invalid temperature value {temperature}")
            pdb.set_trace()
        # return round(float(temp), 1)
        return temp.strip()
    except Exception as e:
        logger.error(f"{e} on temperature value {temperature}")
        pdb.set_trace()


"""
OPAMP NORMALIZERS
"""


def opamp_part_normalizer(part):
    # Part number normalization
    return part.replace(" ", "").upper()


"""
TRANSISTOR NORMALIZERS
"""


def transistor_part_normalizer(part):
    # Part number normalization
    return part.replace(" ", "").upper()


def transistor_temp_normalizer(temperature):
    # On some transistor datasheets the temperature unit is listed as a separate column
    # return round(float(temperature.strip()), 1)
    return temperature.strip()


def polarity_normalizer(polarity):
    if polarity in ["NPN", "PNP"]:
        return polarity
    logger.error(f"Incorrect polarity value {polarity}")
    pdb.set_trace()
    return "N/A"


def dissipation_normalizer(dissipation):
    dissipation = dissipation.strip()
    return str(dissipation.split(" ")[0].strip().replace("-", ""))


def current_normalizer(current):
    current = current.strip()
    try:
        return str(current.split(" ")[0].strip().replace("-", ""))
    except Exception as e:
        logger.error(f"{e} while normalizing current {current}")
        pdb.set_trace()


def voltage_normalizer(voltage):
    voltage = voltage.strip()
    try:
        voltage = voltage.replace("K", "000")
        voltage = voltage.replace("k", "000")
        return voltage.split(" ")[0].strip().replace("-", "")
    except Exception as e:
        logger.error(f"{e} while normalizing voltage {voltage}")
        pdb.set_trace()


def gain_normalizer(gain):
    gain = gain.split("@")[0]
    gain = gain.strip()
    gain = gain.replace(",", "")
    gain = gain.replace("K", "000")
    gain = gain.replace("k", "000")
    return str(gain.split(" ")[0].strip().replace("-", ""))


def old_dev_gain_normalizer(gain):
    return str(abs(int(float(gain))))
