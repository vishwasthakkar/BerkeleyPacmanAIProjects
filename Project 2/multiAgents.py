# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # list that stores the manhattan distances from all the foods to pacman
        distanceToFood = []
        # list that stores the manhattan distances from the ghost to pacman
        distanceToGhost = []
        for g in newGhostStates:
            distanceToGhost.append(util.manhattanDistance(g.getPosition(), newPos))
        for f in newFood.asList():
            distanceToFood.append(util.manhattanDistance(f, newPos))
        # If distance to ghost become zero, then add a heavy negative score
        if min(distanceToGhost) == 0:
            return (-3 * min(distanceToFood)) - (100 * len(distanceToFood)) - 1000
        # If no food available, return huge score
        if len(distanceToFood) == 0:
            return float('inf')
        # Heavy weight to number of food remaining. The more the food is left the lesser the score.
        # Distance to ghost is inversed, means the more the distance between pacman and ghost, the lesser the score
        # becomes negative.
        return (-3 * min(distanceToFood)) - (90 * len(distanceToFood)) - (8 / min(distanceToGhost))


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # Function to generate maxvalue by looking at values of minimizing agent (minAgent).
        def maxAgent(gameState, depth):
            # Get legal actions of pacman.
            legAct = gameState.getLegalActions(0)
            if len(legAct) == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            # If max depth is reached.
            if depth == self.depth:
                return self.evaluationFunction(gameState), None
            # default maximizing value to return
            v = -(float("inf"))
            # default action to return
            Act = None
            for action in legAct:
                # get values of ghosts for each move of pacman
                value = minAgent(gameState.generateSuccessor(0, action), 1, depth)
                # if new value is greater than original value
                if value[0] > v:
                    v = value[0]
                    Act = action
            return v, Act

        # function for minimizing agents. 'Agent' parameter is used to pinpoint a specific ghost
        def minAgent(gameState, agent, depth):
            legAct = gameState.getLegalActions(agent)
            # if no actions are left or terminal node.
            if len(legAct) == 0:
                return self.evaluationFunction(gameState), None
            # default minimizing value
            v = float("inf")
            # default action
            Act = None
            for action in legAct:
                # If last ghost, increase depth and start over
                if agent == gameState.getNumAgents() - 1:
                    value = maxAgent(gameState.generateSuccessor(agent, action), depth + 1)
                else:
                    value = minAgent(gameState.generateSuccessor(agent, action), agent + 1, depth)
                # if new value is smaller than original value
                if value[0] < v:
                    v = value[0]
                    Act = action
            return v, Act

        # Call the maxAgent function for pacman
        x = maxAgent(gameState, 0)[1]
        # return the actions performed by pacman
        return x

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Minimax with Alpha Beta prunning:
        # Function to generate maxvalue by looking at values of minimizing agent (minAgent).
        def maxAgent(gameState, depth, a, b):
            # Get legal actions of pacman.
            legAct = gameState.getLegalActions(0)
            if len(legAct) == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            # If max depth is reached.
            if depth == self.depth:
                return self.evaluationFunction(gameState), None
            # default maximizing value to return
            v = -(float("inf"))
            # default action to return
            Act = None
            for action in legAct:
                # get values of ghosts for each move of pacman
                value = minAgent(gameState.generateSuccessor(0, action), 1, depth, a, b)
                # if new value is greater than original value
                if value[0] > v:
                    v = value[0]
                    Act = action
                if v > b:
                    return v, Act
                a = max(a, v)
            return v, Act

        # function for minimizing agents. 'Agent' parameter is used to pinpoint a specific ghost
        def minAgent(gameState, agent, depth, a, b):
            legAct = gameState.getLegalActions(agent)
            # if no actions are left or terminal node.
            if len(legAct) == 0:
                return self.evaluationFunction(gameState), None
            # default minimizing value
            v = float("inf")
            # default action
            Act = None
            for action in legAct:
                # If last ghost, increase depth and start over
                if agent == gameState.getNumAgents() - 1:
                    value = maxAgent(gameState.generateSuccessor(agent, action), depth + 1, a, b)
                else:
                    value = minAgent(gameState.generateSuccessor(agent, action), agent + 1, depth, a, b)
                # if new value is smaller than original value
                if value[0] < v:
                    v = value[0]
                    Act = action
                if v < a:
                    return v, Act
                b = min(b, v)
            return v, Act

        # Initializing the values of alpha and beta
        a = -(float("inf"))
        b = float("inf")
        x = maxAgent(gameState, 0, a, b)[1]
        return x


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Expectimax is exactly similar to minimax. The only difference is average of random values needs to be done.
        # Function to generate maxvalue by looking at values of minimizing agent (minAgent).
        def maxAgent(gameState, depth):
            # Get legal actions of pacman.
            legAct = gameState.getLegalActions(0)
            if len(legAct) == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            # If max depth is reached.
            if depth == self.depth:
                return self.evaluationFunction(gameState), None
            # default maximizing value to return
            v = -(float("inf"))
            # default action to return
            Act = None
            for action in legAct:
                # get values of ghosts for each move of pacman
                value = minAgent(gameState.generateSuccessor(0, action), 1, depth)
                # if new value is greater than original value
                if value[0] > v:
                    v = value[0]
                    Act = action
            return v, Act

        # 'Agent' parameter is used to pinpoint a specific ghost
        def minAgent(gameState, agent, depth):
            legAct = gameState.getLegalActions(agent)
            # if no actions are left or terminal node.
            if len(legAct) == 0:
                return self.evaluationFunction(gameState), None
            # default minimizing value
            v = 0
            # default action
            Act = None
            for action in legAct:
                # If last ghost, increase depth and start over
                if agent == gameState.getNumAgents() - 1:
                    value = maxAgent(gameState.generateSuccessor(agent, action), depth + 1)
                else:
                    value = minAgent(gameState.generateSuccessor(agent, action), agent + 1, depth)
                # Calculating the average of random values
                temp = value[0]/len(legAct)
                v = v + temp
            return v, Act

        # Call the maxAgent function for pacman
        x = maxAgent(gameState, 0)[1]
        # return the actions performed by pacman
        return x


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      Used a 'score' variable to evaluate the state the pacman is in.
      The evaluation is done by different weightage of different parameters.
    """
    "*** YOUR CODE HERE ***"
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    pacmanPosition = currentGameState.getPacmanPosition()
    capsules = currentGameState.getCapsules()
    score = 0
    distanceToFood = []
    distanceToNormalGhosts = []
    distanceToScaredGhosts = []
    normalGhosts = []
    scaredGhosts = []
    for g in ghosts:
        # If pacman has eaten a capsule.
        if g.scaredTimer:
            scaredGhosts.append(g)
        else:
            normalGhosts.append(g)
    # calculate distances to be used
    for f in food:
        distanceToFood.append(manhattanDistance(pacmanPosition, f))
    for n in normalGhosts:
        distanceToScaredGhosts.append(manhattanDistance(pacmanPosition, n.getPosition()))
    for s in scaredGhosts:
        distanceToScaredGhosts.append(manhattanDistance(pacmanPosition, s.getPosition()))

    # Start calculating the total score:
    # High weightage to number of food remaining
    score += 1.5 * currentGameState.getScore()
    score += -10 * len(food)
    # Very high weight to the capsules yet to be eaten
    score += -20 * len(capsules)
    # Different scores based on distance of pacman to food
    for x in distanceToFood:
        if x < 4:
            score += -1 * x
        if x < 8:
            score += -0.55 * x
        else:
            score += -0.25 * x
    # Very high weight when pacman eats a ghost
    for x in distanceToScaredGhosts:
        if x < 3:
            score += -20 * x
        else:
            score += -10 * x
    # The more the distance to normal or active ghosts the better the state
    for x in distanceToNormalGhosts:
        if x < 3:
            score += 3 * x
        elif x < 7:
            score += 3 * x
        else:
            score += 0.6 * x

    return score


# Abbreviation
better = betterEvaluationFunction

