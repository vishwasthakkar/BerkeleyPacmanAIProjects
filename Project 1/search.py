# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    stack = util.Stack()
    trace = util.Stack()
    explored = []
    stack.push(problem.getStartState())
    trace.push([])
    while True:
        if stack.isEmpty() or trace.isEmpty():
            return []
        a = stack.pop()
        b = trace.pop()
        explored.append(a)
        if problem.isGoalState(a):
            return b
        successors = problem.getSuccessors(a)

        for x in successors:
            if x[0] not in explored:
                stack.push(x[0])
                trace.push(b + [x[1]])


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    frontier = util.Queue()
    trace = util.Queue()
    explored = []
    if problem.isGoalState(problem.getStartState()):
        return []
    frontier.push(problem.getStartState())
    trace.push([])
    while True:
        if frontier.isEmpty() or trace.isEmpty():
            return []
        a = frontier.pop()
        b = trace.pop()
        explored.append(a)
        if problem.isGoalState(a):
            return b
        successors = problem.getSuccessors(a)
        for x in successors:
            if x[0] not in explored and x[0] not in (y[0] for y in frontier.list):
                frontier.push(x[0])
                trace.push(b + [x[1]])
                explored.append(x[0])



def uniformCostSearch(problem):

    frontier = util.PriorityQueue()
    explored = []
    frontier.push((problem.getStartState(), []), 0)
    while True:
        if frontier.isEmpty():
            return []
        node = frontier.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        explored.append(node[0])
        successors = problem.getSuccessors(node[0])
        for child in successors:
            if child[0] not in explored:
                if child[0] not in (y[2][0] for y in frontier.heap):
                    moves = node[1] + [child[1]]
                    cost = problem.getCostOfActions(moves)
                    frontier.push([child[0], moves], cost)
                else:
                    for y in frontier.heap:
                        if child[0] == y[2][0]:
                            cost1 = problem.getCostOfActions(y[2][1])
                    cost2 = problem.getCostOfActions(node[1] + [child[1]])
                    if cost1 > cost2:
                        frontier.push((child[0], node[1]+[child[1]]), cost2)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    def pf(node):
        return problem.getCostOfActions(node[1]) + heuristic(node[0], problem)

    frontier = util.PriorityQueueWithFunction(pf)
    frontier.push((problem.getStartState(), []))
    explored = []
    while True:
        if frontier.isEmpty():
            return []
        node = frontier.pop()
        if node[0] in explored:
            continue
        explored.append(node[0])
        if problem.isGoalState(node[0]):
            return node[1]
        succ = problem.getSuccessors(node[0])
        for child in succ:
            if child[0] not in explored:
                frontier.push((child[0], node[1] + [child[1]]))

    # open = util.PriorityQueueWithFunction(pf)
    # closed = []
    #
    # open.push((problem.getStartState(), []))
    #
    # while not open.isEmpty():
    #     a, path = open.pop()
    #
    #     succ = problem.getSuccessors(a)
    #
    #     if succ:
    #         for child in succ:
    #             if problem.isGoalState(child[0]):
    #                 return child[1]
    #
    #             if child[0] is (y[2][0] for y in open.heap):
    #                 if pf(y[2][0]) < pf(child):
    #                     continue
    #             elif child[0] is (y[2][0] for y in closed):
    #                 if pf(y) < pf(child):
    #                     continue
    #             else:
    #                 open.push((child[0], path + [child[1]]))
    #     closed.append(a);

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
