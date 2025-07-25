import random
from AutomataClasses.PDFA import PDFA


pdfa = PDFA()
a_seen = pdfa.add_node([("a", 9), ("b", 1), ("c", 0.1)], initial_state=True)
b_seen = pdfa.add_node([("a", 1), ("b", 9)])
c_seen = pdfa.add_node([("c", 1)])
pdfa.add_edge(a_seen, "a", a_seen)
pdfa.add_edge(a_seen, "b", b_seen)
pdfa.add_edge(a_seen, "c", c_seen)
pdfa.add_edge(b_seen, "a", a_seen)
pdfa.add_edge(b_seen, "b", b_seen)
pdfa.add_edge(c_seen, "c", c_seen)

string = ""
generator = pdfa.generate_string()
for _ in range(200):
    string += next(generator)
print(string)
