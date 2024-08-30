hyphenated = """
It is a capital mistake to theo-
rize before one has data. Insen-
sibly one begins to twist facts
to suit theories, instead of the-
ories to suit facts.
"""

import re

def dehyphenate(match):
    part1 = match.group(1)
    part2 = match.group(2)

    return f'\n{part1}{part2}'

print(re.sub(r'(\S+)-\n(\S+)', dehyphenate, hyphenated))
