from typing import Any
from AutomataClasses.Functions import Function
from AutomataClasses.SemiAutomaton import SemiAutomaton


class Network(SemiAutomaton):
    def __init__(self, alphabet: set, input_functions: list[Function], semi_automata: list[SemiAutomaton], state_is_final: bool = True,
                 connections: list[list[SemiAutomaton]] | str = "Cascade", update_on_previous_states: bool = True):

        super().__init__(alphabet)

        self.input_functions = input_functions
        self.semi_automata = semi_automata
        self.state_is_final = state_is_final
        self.update_on_previous_states = update_on_previous_states

        if connections == "Cascade":
            self.connections = [self.semi_automata[:i] for i in range(len(semi_automata))]
        else:
            self.connections = connections

    def get_state(self) -> Any:
        if self.state_is_final:
            return self.semi_automata[-1].get_state()
        else:
            return [automaton.get_state() for automaton in self.semi_automata]

    def update(self, token: Any) -> Any:
        super().update(token)

        if self.update_on_previous_states:
            input_values = []
            for input_function, semi_automaton, connections in zip(self.input_functions, self.semi_automata, self.connections):
                semi_automaton_input = tuple([semi_automaton.get_state(), token] + [automaton.get_state() for automaton in connections])
                parsed_semi_automaton_input = input_function.apply(semi_automaton_input)
                input_values.append(parsed_semi_automaton_input)

            for i, semi_automaton in enumerate(self.semi_automata):
                semi_automaton.update(input_values[i])

        else:
            for input_function, semi_automaton, connections in zip(self.input_functions, self.semi_automata, self.connections):
                semi_automaton_input = tuple([semi_automaton.get_state(), token] + [automaton.get_state() for automaton in connections])
                parsed_semi_automaton_input = input_function.apply(semi_automaton_input)
                semi_automaton.update(parsed_semi_automaton_input)

    def reset(self):
        for automaton in self.semi_automata:
            automaton.reset()
