import re


def format_paper(loc):

    with open(loc, 'r') as input:
        file = input.read()

    re.compile(r"\n()")
