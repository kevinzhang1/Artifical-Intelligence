# multiAgents.py
# --------------
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

        "*** YOUR CODE HERE ***"
        l=[]
        ll=newFood.asList()
        for i in ll:
          l.append(util.manhattanDistance(newPos, i))
        if len(l):
          return successorGameState.getScore()+10/min(l)
        else:
          return successorGameState.getScore()

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0)[0]

    def minimax(self, gameState, agentIndex, depth):
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.maxval(gameState, agentIndex, depth)[1]
        else:
            return self.minval(gameState, agentIndex, depth)[1]

    def maxval(self, gameState, agentIndex, depth):
        bestAction = ("max",-float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = max(bestAction,succAction,key=lambda x:x[1])
        return bestAction

    def minval(self, gameState, agentIndex, depth):
        bestAction = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = min(bestAction,succAction,key=lambda x:x[1])
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxvalue(gameState, 1, gameState.getNumAgents()-1, float("-inf"), float("inf"))
        
    def maxvalue(self, state, d, n, a, b):
        v=float("-inf")
        x=0
        if state.isWin() or state.isLose():
          return self.evaluationFunction(state)
        for i in state.getLegalActions(0):
          m=self.minvalue(state.generateSuccessor(0, i), d, 1, n, a, b)
          if m>v:
            v=m
            x=i
          if v>b:
            return v
          a=max(a, v)
        if d<=1:
          return x
        else:
          return v

    def minvalue(self, gameState, d, i, n, a, b):
        v=float("inf")
        x=0
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        for ii in gameState.getLegalActions(i):
          if i==n:
            if self.depth>d:
              x=self.maxvalue(gameState.generateSuccessor(i, ii), d+1, n, a, b)
            else:
              x=self.evaluationFunction(gameState.generateSuccessor(i, ii))
          else:
            x=self.minvalue(gameState.generateSuccessor(i, ii), d, i+1, n, a, b)
          if x<v:
            v=x
          if v<a:
            return v
          b=min(b, v)
        return v

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
        return self.expectimaxagent(gameState, 1, gameState.getNumAgents()-1)

    def expectimaxagent(self, state, d, n):
      v=float("-inf")
      x=0
      if state.isWin() or state.isLose():
          return self.evaluationFunction(state)
      for i in state.getLegalActions(0):
          if v<self.calc(state.generateSuccessor(0, i), d, 1, n):
            v=self.calc(state.generateSuccessor(0, i), d, 1, n)
            x=i
      if d<=1:
        return x
      else:
        return v

    def calc(self, state, d, i, n):   
        v=0
        a=[state.generateSuccessor(i, action) for action in state.getLegalActions(i)]
        b=len(state.getLegalActions(i))
        if state.isWin() or state.isLose():
          return self.evaluationFunction(state)  
        if i==n:
          if self.depth>d:
            for s in a:
              v+=(self.expectimaxagent(s, d+1, n)/b)
          else:
            for s in a:
              v+=(self.evaluationFunction(s)/b)
        else:
          for s in a:
            v+=(self.calc(s, d, i+1, n)/b)
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did> I did what the instructions told me to do which is to write a better evaluation function that evaluates states rather than actions like the reflex agent evaluation function did.
    """
    "*** YOUR CODE HERE ***"
    l=[]
    ll=currentGameState.getFood().asList()
    for i in ll:
      l.append(util.manhattanDistance(currentGameState.getPacmanPosition(), i))
    if len(l):
      if not min(l):
        return currentGameState.getScore()+100
      else:
        return currentGameState.getScore()+10/min(l)
    return currentGameState.getScore()
    ####

# Abbreviation
better = betterEvaluationFunction
