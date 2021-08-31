import numpy as np


class BeliefFilter:
    def __init__(self, boundaries, walls):
        self.boundaries = boundaries
        self.walls = walls
        self.belief = np.ones((boundaries[0], boundaries[1]))

        for wall in walls:
            self.belief[wall] = 0

        self.__normalize()

    def __normalize(self):
        self.belief = self.belief / sum(self.belief.flatten())

    def __get_legal_successors(self, position):
        x, y = position
        successors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [
            p
            for p in successors
            if p not in self.walls
            and p[0] >= 0
            and p[1] >= 0
            and p[0] < self.boundaries[0]
            and p[1] < self.boundaries[1]
        ]

    def time_passes(self):
        new_belief = np.copy(self.belief)
        for (x, y) in np.ndindex(self.belief.shape):
            if self.belief[x, y] == 0:
                continue

            successors = self.__get_legal_successors((x, y))
            new_prob = 0
            for s in successors:
                new_prob += self.belief[s]
            new_belief[x, y] = new_prob / float(len(successors))
        self.belief = new_belief
        self.__normalize()

    def add_evidence(self, evidence, evidence_value=1):
        self.belief[evidence] = evidence_value
        self.__normalize()

    def most_likely(self):
        return np.unravel_index(np.argmax(self.belief), self.belief.shape)
