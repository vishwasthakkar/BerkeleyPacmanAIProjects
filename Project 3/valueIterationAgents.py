# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()

        i = 0
        self.valuesBefore = self.values.copy()

        # Loop until the number of iterations are not complete
        while i < self.iterations:
            # Iterate through each state
            for state in mdp.getStates():
                # if the state encountered is not a terminal, compute the QValues
                if not mdp.isTerminal(state):
                    valueOfAnAction = -99999
                    # For each possible action for a state:
                    for action in mdp.getPossibleActions(state):
                        valueOfAnAction = max(valueOfAnAction, self.computeQValueFromValues(state, action))
                    self.values[state] = valueOfAnAction
            self.valuesBefore = self.values.copy()
            # print "\n\n\nValuesBefore", self.valuesBefore
            # print "\n\n\nValues", self.values
            i += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q = 0
        # For an action get the possible next states and transition probabilities
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        # iterate through each transition from a given state to a possible next state
        # for t in transitions:
        i = 0
        while i < len(transitions):
            q = q + (transitions[i][1] * (self.discount * self.valuesBefore[transitions[i][0]] +
                                          self.mdp.getReward(state, action, transitions[i][0])))
            i += 1
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actionTaken = None
        Value = -99999

        i = 0
        while i < len(self.mdp.getPossibleActions(state)):
            qValue = self.computeQValueFromValues(state, self.mdp.getPossibleActions(state)[i])
            if (qValue > Value):
                Value = qValue
                actionTaken = self.mdp.getPossibleActions(state)[i]
            i += 1
        return actionTaken


def getPolicy(self, state):
    return self.computeActionFromValues(state)


def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.computeActionFromValues(state)


def getQValue(self, state, action):
    return self.computeQValueFromValues(state, action)
