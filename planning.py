from game import Actions, Grid
import util, numpy as np


def makePlan(gameState, start, goal, h):
    fScoreStart = h(start, goal)

    walls = gameState.getWalls()
    width = walls.width
    height = walls.height

    fringe = util.PriorityQueue()
    fringe.push(start, fScoreStart)

    cameFrom = Grid(width, height)
    gScore = Grid(width, height)
    fScore = Grid(width, height)
    openSet = set([start])

    for x in range(width):
        for y in range(height):
            cameFrom[x][y] = None
            gScore[x][y] = np.inf
            fScore[x][y] = np.inf

    fScore[start[0]][start[1]] = fScoreStart
    gScore[start[0]][start[1]] = 0

    while not fringe.isEmpty():
        current = fringe.pop()

        if current == goal:
            return reconstructPath(cameFrom, goal)

        for nb in Actions.getLegalNeighbors(current, walls):
            tentativeGScore = gScore[current[0]][current[1]] + 1
            if tentativeGScore < gScore[nb[0]][nb[1]]:
                cameFrom[nb[0]][nb[1]] = current
                gScore[nb[0]][nb[1]] = tentativeGScore
                fScore[nb[0]][nb[1]] = gScore[nb[0]][nb[1]] + h(nb, goal)

                if nb not in openSet:
                    openSet.add(nb)
                    fringe.push(nb, fScore[nb[0]][nb[1]])


def reconstructPath(cameFrom, goal):
    current = goal
    path = [current]
    while current != None:
        current = cameFrom[current[0]][current[1]]
        path.append(current)
    path.reverse()
    return path[1:]


def followPlan(myPosition, plan, gameState):
    try:
        goal = plan[1]
    except IndexError:
        goal = plan[0]

    dx = goal[0] - myPosition[0]
    dy = goal[1] - myPosition[1]

    return Actions.vectorToDirection((dx, dy))


def nearestPosition(myPosition, positions):
    return min(positions, key=lambda x: util.manhattanDistance(myPosition, x))


def applyField(origin, attractors, deflectors):
    return max(
        attractors,
        key=lambda y: util.manhattanDistance(
            max(
                deflectors,
                key=lambda x: util.manhattanDistance(origin, x),
            ),
            y,
        ),
    )
