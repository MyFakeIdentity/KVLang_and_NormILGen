import random

import numpy as np


class Recogniser:
    def __init__(self, window_size, view_size, depth):
        self.window_size = window_size
        self.view_size = view_size
        self.depth = depth

        self.memory = np.zeros((self.depth, self.window_size + 1), np.int32)

    def update(self, symbol):
        self.memory = np.roll(self.memory, -1)
        self.memory[0, -1] = symbol

        for layer in range(1, self.depth):
            segment = slice(max(layer - self.view_size, 0), layer)
            num_ones = np.sum(self.memory[segment, :-1])
            # num_values = (segment.stop - segment.start) * self.window_size

            # if num_ones > num_values / 2:
            if num_ones % 2 == 0:
                self.memory[layer, -1] = 0
            else:
                self.memory[layer, -1] = 1

        # print(self.memory)

        return self.memory[-1, -1] == 1


recogniser = Recogniser(4, 3, 7)

while True:
    output = None
    string = []
    for _ in range(random.randint(10, 50)):
        val = random.randint(0, 1)
        string.append(val)
        output = recogniser.update(val)
    print("".join(map(str, string)), output)


# Language(h): {{0^n_i 1^n_i}* | forall i, n_i <= h}. Increment counter and decrement counter check counter = 0 when moving from 1 to a 0.

# Language(h): 1's appear in groups of powers of 2 where size is <= h. Increment counter and check if only 1 bit set.

# Language(k, v): [(key, value), ...]   every time a specific key occurs the same value must follow, where |key| = k, |value| = v. Store
# every seen key value pair (possible due to finite number of keys) store currently seen key and value compair to previously seen. No real
# way to control density (what happens if alphabet size is increased).
# Depth via history of values (e.g. function over 5 most recent values for each key).
#   Previous n occurrences of each key have majority for each bit (can be done via counter).
#   Previous n occurrences of each key have either all = 1 or all = 0 for each bit (can be done via counter).
#   Previous n with bitwise or (alternatives: least n/2 set per bit, ...).
# Depth via key -> key -> value, etc.
#   Hyper keys denoted by a value before (key level - key - value) where value points to a key of 1 level lower. All base values for max keys
#       must be the same.

# Language(h): {{0^n_i 1^(2^n_i)}* | forall i, n_i <= h}.

# Binary addition.
# What other functions would give us a better sense of depth.
