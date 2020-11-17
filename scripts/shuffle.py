import sys
import random

random.seed(17)

for filename in sys.argv[1:]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
