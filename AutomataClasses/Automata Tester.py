import random
from AutomataClasses.Network import Network
from AutomataClasses.FlipFlop import FlipFlop
from Functions import Function


# Invalidated as network components read their own state as well.
test = Network({"a", "b", "c"}, [
    Function({("a",): "Read", ("b",): "Set", ("c",): "Reset"}),
    Function({("a", "Low"): "Read", ("a", "High"): "Set", ("b", "Low"): "Set", ("b", "High"): "Reset", ("c", "Low"): "Read", ("c", "High"): "Set"}),
    Function({("a", "Low", "Low"): "Set", ("a", "High", "Low"): "Reset", ("b", "Low", "Low"): "Reset", ("b", "High", "Low"): "Reset", ("c", "Low", "Low"): "Read", ("c", "High", "Low"): "Set",
              ("a", "Low", "High"): "Read", ("a", "High", "High"): "Read", ("b", "Low", "High"): "Set", ("b", "High", "High"): "Set", ("c", "Low", "High"): "Read", ("c", "High", "High"): "Read"})
], [
    FlipFlop(),
    FlipFlop(),
    FlipFlop(),
])

for _ in range(100):
    token = random.choice(["a", "b", "c"])
    test.update(token)
    print(token, test.get_state(), [semi_automaton.get_state() for semi_automaton in test.semi_automata])
