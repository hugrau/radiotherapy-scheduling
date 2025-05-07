from unidecode import unidecode
import re


def apply_unidecode(element):
    """
    Apply unidecode to element if is of type str.
    :param element:
    :return: element with replaced unidecode
    """
    if isinstance(element, str):
        return unidecode(element)
    return element


def apply_replace(element):
    """
    Apply "replace" to element if is of type str.
    :param element:
    :return: element with replacement
    """
    if isinstance(element, str):
        return element.replace("œ", "oe")
    return element


def apply_lower(element):
    """
    Apply "lower" to element if is of type str.
    :param element:
    :return: element with lower case.
    """
    if isinstance(element, str):
        return element.lower()
    return element


def separate_letters_and_digits(string):
    # Use regular expression to separate letters and digits.
    letters = re.findall("[a-zA-Z]+", string)
    digits = re.findall("\d+", string)

    # Join letters and digits
    letters_joined = "".join(letters)
    digits_joined = "".join(digits)

    return " ".join([letters_joined, digits_joined])


def insert_space_before_numbers(text):
    # Use regular expression to find instances of digits followed by characters
    # and insert a space between them.
    return re.sub(r"([a-zA-Z])(\d)|(\d)([a-zA-Z])", r"\1\3 \2\4", text)
