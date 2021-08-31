# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from communication import Communication
import random
from util import Queue
from distanceCalculator import manhattanDistance
from captureAgents import CaptureAgent
from beliefFilter import BeliefFilter
from planning import makePlan, followPlan, nearestPosition

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed, first="Gluttony", second="Gluttony"):
    comm = Communication()

    firstAgent = eval(first)(firstIndex)
    firstAgent.addComm(comm)

    secondAgent = eval(second)(secondIndex)
    secondAgent.addComm(comm)

    return [firstAgent, secondAgent]


##########
# Agents #
##########


class Gluttony(CaptureAgent):
    hasEaten = 0
    belief = []
    teamMates = []
    opponents = []

    currentBehaviour = None
    state = "start"

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        if self.red:
            self.teamMates = gameState.getRedTeamIndices()
            self.opponents = gameState.getBlueTeamIndices()
        else:
            self.teamMates = gameState.getBlueTeamIndices()
            self.opponents = gameState.getRedTeamIndices()

        self.num_opponents = len(self.opponents)

        self.walls = gameState.getWalls().asList()

        self.height = gameState.getWalls().height
        self.width = gameState.getWalls().width

        for _ in range(self.num_opponents):
            self.belief.append(BeliefFilter((32, 16), self.walls))

    def addComm(self, comm):
        self.messages = comm

    def nearest_enemy_to(self, position):
        nearestIdx = -1
        nearest = None
        nearestDistance = 999
        for idx in range(self.num_opponents):
            likely = self.belief[idx].most_likely()
            d = manhattanDistance(position, likely)
            if d < nearestDistance:
                nearestDistance = d
                nearestIdx = idx
                nearest = likely
        return nearestIdx, nearest

    def __food_disparity(self):
        try:
            before = self.getPreviousObservation()
            assert before != None
        except (IndexError, AssertionError):
            return set()

        now = self.getCurrentObservation()

        if self.red:
            foodBefore = set(before.getRedFood().asList())
            foodNow = set(now.getRedFood().asList())
            return foodBefore - foodNow
        else:
            foodBefore = set(before.getBlueFood().asList())
            foodNow = set(now.getBlueFood().asList())
            return foodBefore - foodNow

    def __capsule_disparity(self):
        try:
            before = self.getPreviousObservation()
            assert before != None
        except (IndexError, AssertionError):
            return set()

        now = self.getCurrentObservation()

        if self.red:
            capsulesBefore = set(before.getRedCapsules())
            capsulesNow = set(now.getRedCapsules())
            return capsulesBefore - capsulesNow
        else:
            capsulesBefore = set(before.getBlueCapsules())
            capsulesNow = set(now.getBlueCapsules())
            return capsulesBefore - capsulesNow

    def __coloredIndex(self, idx):
        if idx == 0:
            return [1, 0, 0]
        elif idx == 1:
            return [0, 1, 0]

    def update_beliefs(self, gameState):
        foodEaten = self.__food_disparity()
        capsuleEaten = self.__capsule_disparity()

        for food in foodEaten | capsuleEaten:
            culprit, _ = self.nearest_enemy_to(food)
            self.belief[culprit].add_evidence(food)

        for idx in range(self.num_opponents):
            opponent = self.opponents[idx]
            opponent_position = gameState.getAgentPosition(opponent)
            my_position = gameState.getAgentPosition(self.index)

            if opponent_position is not None:
                self.belief[idx].add_evidence(opponent_position)
            else:
                self.belief[idx].add_evidence(my_position, 0)
                self.belief[idx].time_passes()

    def debug_beliefs(self):
        self.debugClear()

        for i in range(self.num_opponents):
            opponent_should_be = self.belief[i].most_likely()
            self.debugDraw([opponent_should_be], self.__coloredIndex(i))

    def patrolBehaviour(self, myPosition):
        if random.uniform(0, 1) > 0.5:
            return (18, 7)
        else:
            return (18, 9)

    def defensiveBehaviour(self, myPosition):
        _, nearby = self.nearest_enemy_to(myPosition)

        # If any enemy is in our side, this ruthless ghost will hunt him down
        # to the last drop of blood >:D
        if nearby[0] > 16:
            return nearby
        else:
            return self.patrolBehaviour(myPosition)

    def retreatBehaviour(self, myPosition):
        opponentIdxs = self.opponents
        opponentPositions = [self.belief[idx].most_likely() for idx in opponentIdxs]

        viable = [(18, y) for y in range(self.height) if (18, y) not in self.walls]
        distance = [manhattanDistance(myPosition, x) for x in viable]

        dangers = []
        for target in viable:
            danger = 0.0
            for opponent in opponentPositions:
                danger += manhattanDistance(opponent, target)
            dangers.append(danger / float(self.num_opponents))

        bestRetreatRoute = -1
        bestRetreatValue = 999999
        for idx in range(len(distance)):
            if dangers[idx] > 0:
                value = distance[idx] / dangers[idx]
                if value < bestRetreatValue or bestRetreatValue == -1:
                    bestRetreatRoute = idx
                    bestRetreatValue = value
        return viable[bestRetreatRoute]

    def incursionBehaviour(self, myPosition):
        gameState = self.getCurrentObservation()

        opponentIdxs = self.opponents
        opponentPositions = [self.belief[idx].most_likely() for idx in opponentIdxs]

        if self.red:
            foods = gameState.getBlueFood().asList()
        else:
            foods = gameState.getRedFood().asList()

        safestFood = None
        safestFoodSafeness = 0
        for food in foods:
            safeness = 0.0
            for opponent in opponentPositions:
                safeness += manhattanDistance(food, opponent)

            if safeness > safestFoodSafeness or safestFood is None:
                safestFood = food
                safestFoodSafeness = safeness

        return safestFood

    def scaredBehaviour(self, myPosition):
        opponentIdxs = self.opponents
        opponentPositions = [self.belief[idx].most_likely() for idx in opponentIdxs]

        for opponent in opponentPositions:
            if manhattanDistance(myPosition, opponent) < 6:
                return self.retreatBehaviour(myPosition)

    def heroBehaviour(self, myPosition):
        gameState = self.getCurrentObservation()

        if self.red:
            capsules = gameState.getBlueCapsules()
        else:
            capsules = gameState.getRedCapsules()
        if len(capsules) > 0:
            return nearestPosition(myPosition, capsules)
        else:
            return self.scaredBehaviour(myPosition)

    def envEvents(self, myPosition):
        # Stores as a class attribute whether any food was eaten this turn
        # by which player and whether any capsules were eaten by which player.
        # Also if any player died.
        try:
            before = self.getPreviousObservation()
            assert before != None
        except (IndexError, AssertionError):
            return {
                "foodEaten": 0,
                "capsulesEaten": 0,
                "didIeatCapsule": False,
                "didIeatFood": False,
                "atCenter": False,
            }

        now = self.getCurrentObservation()
        opponentCapsulesBefore = set(before.getBlueCapsules())
        opponentCapsulesNow = set(now.getBlueCapsules())
        opponentFoodBefore = set(before.getBlueFood().asList())
        opponentFoodNow = set(now.getBlueFood().asList())

        myFoodBefore = set(before.getRedFood().asList())
        myFoodNow = set(now.getRedFood().asList())
        myCapsulesBefore = set(before.getRedCapsules())
        myCapsulesNow = set(now.getRedCapsules())

        capsulesWeAte = opponentCapsulesNow - opponentCapsulesBefore
        foodWeAte = opponentFoodBefore - opponentFoodNow

        capsulesOpponnentAte = myCapsulesBefore - myCapsulesNow
        foodOpponnentAte = myFoodBefore - myFoodNow

        didIEatCapsule = myPosition in capsulesWeAte
        didIEatFood = myPosition in foodWeAte

        return {
            "foodEaten": (len(foodWeAte), len(foodOpponnentAte)),
            "capsulesEaten": (len(capsulesWeAte), len(capsulesOpponnentAte)),
            "didIeatCapsule": didIEatCapsule,
            "didIeatFood": didIEatFood,
            "atCenter": manhattanDistance(myPosition, (18, 7)) < 6,
        }

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        self.update_beliefs(gameState)
        self.debug_beliefs()

        myPosition = gameState.getAgentPosition(self.index)
        evts = self.envEvents(myPosition)
        messages = self.messages.pending(self.index)

        if self.state == "start":
            if not evts["atCenter"]:
                self.currentBehaviour = self.patrolBehaviour
                self.state = "fellingGood"
        elif self.state == "fellingGood" and evts["atCenter"]:
            if self.index == 1:
                self.currentBehaviour = self.defensiveBehaviour
                self.messages.say("defense", self.index)
            elif self.index == 3:
                self.currentBehaviour = self.incursionBehaviour
                self.messages.say("incursion", self.index)
        elif self.state == "incursion" and evts["didIeatFood"]:
            if evts["didIeatFood"]:
                self.currentBehaviour = self.retreatBehaviour
                self.state = "retreat"
            elif "hero" in messages:
                self.currentBehaviour = self.defensiveBehaviour
                self.state = "defense"
        elif self.state == "retreat" and myPosition[0] > 16:
            self.currentBehaviour = self.incursionBehaviour
            self.state = "incursion"
        elif self.state == "hero":
            if evts["foodEaten"][1] > 2:
                self.currentBehaviour = self.defensiveBehaviour
                self.state = "defense"

        feelingHero = random.random() < 0.05
        if feelingHero:
            self.messages.say("hero", self.index)
            self.currentBehaviour = self.heroBehaviour

        self.messages.clear(self.index)
        goal = self.currentBehaviour(myPosition)

        if goal == None:
            goal = self.defensiveBehaviour(myPosition)

        plan = makePlan(gameState, myPosition, goal, manhattanDistance)
        return followPlan(myPosition, plan, gameState)