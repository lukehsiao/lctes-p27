import csv
import logging
import os
import pdb

logger = logging.getLogger(__name__)


# A list of known manufacturers
VALID_MANUF = [
    "NXP Semiconductors",
    "STMicroelectronics",
    "ON Semiconductor",
    "Fairchild",
    "AUK",
    "MCC",
    "Taiwan",
    "Bourns",
    "Central Semiconductor",
    "Diodes Incorporated",
    "Diotec",
    "Fairchild Semiconductor",
    "Infineon",
    "JCST",
    "KEC",
    "Vishay",
    "Lite-on",
    "Vishay/Lite-on",
    "Minilogic",
    "Motorola",
    "Micro Commercial Components",
    "Microsemi",
    "Philips",
    "Panjit",
    "Rectron",
    "Secos",
    "Siemens",
    "Sanken",
    "Tak Cheong",
    "TT Electronics",
    "UTC",
    "General",
    "Weitron",
    "Mouser",
    "Rohm Semiconductor",
    "Aeroflex",
]


# A list of manufacturers to be filtered that we know do not appear in dev and test
# gold labels and thus are not in the `docs` dict returned by `get_docs()`
INVALID_MANUF = [
    "Nexperia USA Inc.",
    "Renesas Electronics America",
    "Comchip Technology",
    "Toshiba Semiconductor and Storage",
    "Panasonic Electronic Components",
    "WeEn Semiconductors",
    "M/A-Com Technology Solutions",
    "Texas Instruments",
    "Parallax Inc.",
    "SANYO Semiconductor (U.S.A) Corporation",
]


# A dictionary of common manufacturer acronyms that maps to a value
# in VALID_MANUF
# TODO: Vishay/Lite-on is recognized as it's own manuf here, should we
# combine Vishay and Lite-on into that manuf?
MANUF = {
    "Micro Commercial Co": "Micro Commercial Components",
    "On Semiconductor": "ON Semiconductor",
    "Central": "Central Semiconductor",
    "Central Semiconductor Corp": "Central Semiconductor",
    "ST": "STMicroelectronics",
    "NXP": "NXP Semiconductors",
    "Rohm": "Rohm Semiconductor",
    "Microsemi Corporation": "Microsemi",
    "TT Electronics/Optek Technology": "TT Electronics",
    "Taiwan Semiconductor Corporation": "Taiwan",
    "Infineon Technologies": "Infineon",
    "MICROSS/On Semiconductor": "ON Semiconductor",
    "NXP USA Inc.": "NXP Semiconductors",
    "Vishay Semiconductor Diodes Division": "Vishay",
    "Bourns Inc.": "Bourns",
}


"""
TRANSISTOR PREPROCESSORS:
"""


def preprocess_url(url):
    # Takes in a URL and returns it if it is valid
    if url == "-":
        return "N/A"
    return url  # We use the PDFs URL as the filename as we cannot ensure
    # standard datasheet filenames: allows for convenient file lookup


def preprocess_manuf(manuf):
    if manuf in VALID_MANUF or manuf == "N/A":
        return manuf
    elif manuf in MANUF:
        return MANUF[manuf]
    elif manuf in INVALID_MANUF:
        return "N/A"
    else:
        logger.error(f"Invalid manuf {manuf}.")
        pdb.set_trace()


def get_docs(
    dev_gold=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../dev/dev_gold.csv"
    ),
    test_gold=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../test/test_gold.csv"
    ),
    debug=False,
):
    """Reads in the dev and test gold files and returns a dictionary that can be used
    to find a filename through manuf and part number."""
    duplicate_filenames = set()
    docs = {}
    with open(dev_gold, "r") as dev, open(test_gold, "r") as test:
        devreader = csv.reader(dev)
        testreader = csv.reader(test)

        # Read in dev gold
        for line in devreader:
            (filename, manuf, part, attr, val) = line
            manuf = preprocess_manuf(manuf)
            # Skip invalid manuf (i.e. manuf that appear in INVALID_MANUF)
            if manuf == "N/A" or manuf is None:
                continue
            if manuf in docs:
                if part in docs[manuf]:
                    if docs[manuf][part] != filename:
                        logger.warning(
                            f"Filenames {docs[manuf][part]} and "
                            + f"{filename} do not match, using {filename}."
                        )
                        duplicate_filenames.add((filename, docs[manuf][part]))
                # Only use the last seen filename
                docs[manuf][part] = filename
            else:
                docs[manuf] = {part: filename}

        # Read in test gold
        for line in testreader:
            (filename, manuf, part, attr, val) = line
            manuf = preprocess_manuf(manuf)
            # Skip invalid manuf (i.e. manuf that appear in INVALID_MANUF)
            if manuf == "N/A" or manuf is None:
                continue
            if manuf in docs:
                if part in docs[manuf]:
                    if docs[manuf][part] != filename:
                        logger.warning(
                            f"Filenames {docs[manuf][part]} and "
                            + f"{filename} do not match, using {filename}."
                        )
                        duplicate_filenames.append((filename, docs[manuf][part]))
                # Only use the last seen filename
                docs[manuf][part] = filename
            else:
                docs[manuf] = {part: filename}

        if debug:
            if len(duplicate_filenames) != 0:
                logger.error(f"There were {len(duplicate_filenames)} duplicate files:")
                logger.error(f"Duplicate filenames {duplicate_filenames}")
                pdb.set_trace()
        if len(docs) != 0 and docs is not None:
            return docs

        else:
            logger.error(f"Gold document reference is empty.")
            pdb.set_trace()


def preprocess_doc(manuf, part, url, docformat="standard", docs=get_docs(debug=True)):
    """Returns the filename of a given document by cross referencing our gold data"""
    if docformat == "url":
        if url == "-" or url is None:
            return "N/A"
        elif url.strip().endswith(".pdf"):
            return url.split("/")[-1].strip(".pdf")
        elif url.strip().endswith(".PDF"):
            return url.split("/")[-1].strip(".PDF")
        else:
            logger.warning(f"Couldn't get filename for {url}, using URL.")
            return url.strip()

    elif docformat == "standard":
        # Get filename by cross referencing with our gold labels
        try:
            manuf = preprocess_manuf(manuf)
            # Skip invalid manuf that we know are not in `docs`
            # (i.e. manuf that appear in INVALID_MANUF)
            if manuf == "N/A" or manuf is None:
                return "N/A"
            # Otherwise, get the filename from doc_name
            doc_name = docs[manuf][part]
            return doc_name.replace(".pdf", "").replace(".PDF", "").strip()
        except Exception as e:
            if manuf not in docs:
                logger.error(f"Manuf {manuf} not found in {[i for i in docs]}.")
                pdb.set_trace()
            logger.warning(
                f"{e} while fetching doc_name for {manuf}: {part}, skipping."
            )
            return "N/A"

    else:
        logger.error(f"Invalid filename format {format}, using standard.")
        return preprocess_doc(manuf, part, url, docs=docs)


def add_space(type, value):
    value = value.strip()
    if type == "current":
        if value.endswith("nA"):
            return value.replace("nA", " nA")
        elif value.endswith("mA"):
            return value.replace("mA", " mA")
        # Account for exception on line 75 of ffe00114_3.csv
        # See: https://www.digikey.com/products/en?keywords=2SD2704KT146TR-ND
        elif value.endswith("ma"):
            return value.replace("ma", " mA")
        elif value.endswith("A"):
            return value.replace("A", " A")
        else:
            logger.warning(f"Invalid {type} {value}")
            pdb.set_trace()
    elif type == "voltage":
        if value.endswith("mV"):
            return value.replace("mV", " mV")
        elif value.endswith("V"):
            return value.replace("V", " V")
        else:
            logger.warning(f"Invalid {type} {value}")
            pdb.set_trace()
    elif type == "frequency":
        if value.endswith("MHz"):
            return value.replace("MHz", " MHz")
        elif value.endswith("GHz"):
            return value.replace("GHZ", " GHz")
        elif value.endswith("kHz"):
            return value.replace("kHz", " kHz")
        else:
            logger.warning(f"Invalid {type} {value}")
            pdb.set_trace()
    elif type == "power":
        if value.endswith("mW"):
            return value.replace("mW", " mW")
        elif value.endswith("W"):
            return value.replace("W", " W")
        else:
            logger.warning(f"Invalid {type} {value}")
            pdb.set_trace()


def preprocess_dc_gain_min(gain):
    # Takes in a dc_gain_min with Digikey's standard condition syntax
    # (i.e. 200 @ 2mA, 5V)
    # And returns a tuple containing (dc_gain, Ic, Vce) with units
    if gain == "-":
        return "N/A"
    try:
        (dc_gain, conditions) = gain.split("@")
        dc_gain = dc_gain.strip()
        conditions = conditions.strip()
        # Here we also return implied values (found in conditions)
        # (i.e. dc_gain_min @ supply_current) <-- We can extract supply_current
        (implied_supply_current, implied_ce_v_max) = conditions.split(",")

        # Add space between value and unit
        # Account for unit exception on line 163 of ffe00114_15.csv
        # See: https://www.digikey.com/products/en?keywords=2SC2922
        if implied_supply_current.endswith("V"):
            implied_supply_current = add_space("voltage", implied_supply_current)
        else:
            implied_supply_current = add_space("current", implied_supply_current)
        # Account for unit exception on line 262 of ffe00114_38.csv
        # See: https://www.digikey.com/products/en?keywords=BD249C-S-ND
        if implied_ce_v_max.endswith("A"):
            implied_ce_v_max = add_space("current", implied_ce_v_max)
        else:
            implied_ce_v_max = add_space("voltage", implied_ce_v_max)

        # Return final values (formatted as the value, a space, and the unit)
        return (dc_gain, implied_supply_current, implied_ce_v_max)
    except Exception as e:
        logger.error(
            f"{e} while preprocessing dc current gain min: {gain}" + "returning N/A."
        )
        pdb.set_trace()  # TODO: Do we just want to return N/A or pdb?
        return "N/A"


def preprocess_vce_saturation_max(voltage):
    # Takes in a vce_saturation_max with Digikey's standard condition syntax
    # (i.e. 600mV @ 5mA, 100mA)
    # And returns a tuple containing (vce_sat_max, Ib, Ic) with units
    if voltage == "-":
        return "N/A"
    try:
        (vce_sat_max, conditions) = voltage.split("@")
        conditions = conditions.strip()

        # Here we also return implied values (found in conditions)
        # (i.e. dc_gain_min @ supply_current) <-- We can extract supply_current
        (implied_base_current, implied_supply_current) = conditions.split(", ")

        # Add space between the value and unit
        # Account for invalid unit discrepancy in ffe00114_0.csv at line 486
        # See: https://www.digikey.com/products/en?keywords=MJE18008G
        if implied_supply_current.endswith("V"):
            implied_supply_current = add_space("voltage", implied_supply_current)
        else:
            implied_supply_current = add_space("current", implied_supply_current)

        implied_base_current = add_space("current", implied_base_current)
        vce_sat_max = add_space("voltage", vce_sat_max)

        # Return final set
        return (vce_sat_max, implied_base_current, implied_supply_current)
    except Exception as e:
        logger.error(
            f"{e} while preprocessing collector emitter saturation"
            + f" voltage max: {voltage} returning N/A."
        )
        pdb.set_trace()  # TODO: Do we just want to return N/A or pdb?
        return "N/A"


def preprocess_ce_v_max(voltage):
    # Takes in a ce_v_max with Digikey's standard condition syntax
    # (i.e. 65V)
    # Checks if it's valid and returns the stripped value
    if voltage == "-":
        return "N/A"
    try:
        return add_space("voltage", voltage)
    # TODO: Why do I even have this try and except here??
    except Exception as e:
        logger.error(
            f"{e} while preprocessing collector emitter"
            + f" voltage max: {voltage} returning N/A."
        )
        pdb.set_trace()  # TODO: Do we just want to return N/A or pdb?
        return "N/A"


def preprocess_c_current_max(current):
    # Takes in a c_current_max with Digikey's standard condition syntax
    # (i.e. 100mA)
    # Checks if it's valid and returns the stripped value
    if current == "-":
        return "N/A"
    # Add space between the value and unit
    return add_space("current", current)


def preprocess_polarity(polarity):
    # Takes in a polarity (i.e. NPN) and returns it if it is valid
    if polarity in ["NPN", "PNP"]:
        return polarity
    elif polarity == "-":
        return "N/A"

    # Handle cases where polarity includes extra conditions
    extras = [
        " - Darlington",
        " - Avalanche Mode",
        " + Zener",
        " - Trilinton, Zener Clamp",
        " + Diode (Isolated)",
        " + Zener Diode (Isolated)",
        " - Emitter Switched Bipolar",
    ]
    for extra in extras:
        if polarity.endswith(extra):
            if polarity.strip(extra) in ["NPN", "PNP"]:
                return polarity.strip(extra)

    # Handle special cases
    if polarity == "2 PNP (Dual)":
        return "PNP"

    logger.error(f"Invalid polarity {polarity}")
    pdb.set_trace()
    return "N/A"


def preprocess_c_current_cutoff_max(current):
    # Takes in a c_current_cutoff_max with Digikey's standard condition syntax
    # (i.e. 15nA (ICBO))
    # Checks if it's valid and returns the stripped value
    if current == "-":
        return "N/A"
    current = current.strip("(ICBO)")
    # Add space between value and unit
    return add_space("current", current)


def preprocess_pwr_max(power):
    if power == "-":
        return "N/A"
    return add_space("power", power)


def preprocess_freq_transition(freq):
    if freq == "-":
        return "N/A"
    # Add space between value and unit
    return add_space("frequency", freq)


"""
OPAMP PREPROCESSORS:
TODO: These have not been fully vetted yet
"""


def preprocess_gbp(typ_gpb):
    if typ_gpb == "-":
        return "N/A"
    else:
        return typ_gpb


def preprocess_supply_current(current):
    supply_current = current.replace("Â", "").replace("µ", "u")

    # sometimes digikey reports a random MAX.
    supply_current = supply_current.replace("(Max)", "").strip()

    if supply_current == "-":
        return "N/A"
    else:
        return add_space("current", supply_current)


def preprocess_operating_voltage(voltage):
    # handle strings like:
    #   2.4 V ~ 6 V
    #   4.5 V ~ 36 V, Â±2.25 V ~ 18 V
    #   10 V ~ 36 V, Â±5 V ~ 18 V
    #   4.75 V ~ 5.25 V, Â±2.38 V ~ 2.63 V
    if voltage == "-":
        return ("N/A", "N/A")

    op_volt = voltage.replace("Â", "")
    if "~" not in op_volt and "," in op_volt:
        op_volt = op_volt.replace(",", "~")
    elif "~" not in op_volt:  # for when only a single value is reported
        op_volt = " ~ ".join([op_volt, op_volt])
    ranges = [r.strip() for r in op_volt.split(",")]
    min_set = set()
    max_set = set()
    for r in ranges:
        try:
            (min_val, max_val) = [val.strip() for val in r.split("~")]
            if "/" in min_val or "/" in max_val:
                continue  # -0.9 V/+1.3 V-0.9 V/+1.3 V in ffe002af_13.csv
            if " " not in min_val:
                min_val = min_val[:-1] + " " + min_val[-1:]
            if " " not in max_val:
                max_val = max_val[:-1] + " " + max_val[-1:]
            min_set.add(add_space("voltage", min_val))
            max_set.add(add_space("voltage", max_val))
        except ValueError as e:
            logger.error(
                f"{e} while preprocessing operating voltage: {voltage}"
                + "returning N/A."
            )
            pdb.set_trace()
            return ("N/A", "N/A")

    return (";".join(min_set), ";".join(max_set))


def preprocess_operating_temp(temperature):
    if temperature == "-" or temperature is None:
        return ("N/A", "N/A")
    # handle strings like:
    #   -20Â°C ~ 75Â°C
    op_temp = temperature.replace("Â", "").replace("°", " ")

    # Deal with temperatures like: 150Â°C (TJ)
    if "~" not in op_temp:  # For values like: 150 C (TJ)
        return (op_temp.strip("(TJ)"), op_temp.strip("(TJ)"))

    try:
        (min_temp, max_temp) = [val.strip() for val in op_temp.split("~")]
        # Add a space in between value and unit:
        (min_temp, max_temp) = (min_temp.strip("(TJ)"), max_temp.strip("(TJ)"))
        return (min_temp, max_temp)
    except ValueError as e:
        logger.error(
            f"{e} while preprocessing operating temperature range: "
            + f"{temperature} returning N/A."
        )
        pdb.set_trace()
        return ("N/A", "N/A")
