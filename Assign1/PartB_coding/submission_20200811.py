## ID: 20200811 NAME: Maeng, Chanyoung
######################################################################################
# Problem 2a
# minimax value of the root node: 5
# pruned edges: h, m
######################################################################################

from pacman import GameState
from symbol import eval_input
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
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
    
    #define 1. max function for pacman 2. min function for ghosts and move agents from pacman to n-th ghost : 1 depth
    # for pacman agent who should maximize
    def max_valuing(depth, state):      
      if self.depth == depth or state.isLose() or state.isWin():  
       return self.evaluationFunction(state)
    
      val = float('-inf')
      legalActions = state.getLegalActions(0)
      for legalAction in legalActions:
        val = max(val, min_valuing(depth, state.generateSuccessor(0, legalAction), 1))
      return val

    # for ghost agent who should minimize
    def min_valuing(depth, state, agentIndex):
      if self.depth == depth or state.isLose() or state.isWin():  
        return self.evaluationFunction(state)

      val = float("inf")
      legalActions = state.getLegalActions(agentIndex)
      #checking whether this is final ghost or not
      if agentIndex == state.getNumAgents()-1:  #final ghost : next turn is for pacman
        for legalAction in legalActions:
          val = min(val, max_valuing(depth+1, state.generateSuccessor(agentIndex, legalAction)))
      else:
        for legalAction in legalActions:
          val = min(val, min_valuing(depth, state.generateSuccessor(agentIndex, legalAction), agentIndex + 1))
      return val        

    #moving pacman : find the acting which makes maximum value of successors' value 
    max_val = float('-inf')
    acting = Directions.STOP
    legalActions = gameState.getLegalActions(0)
    for legalAction in legalActions:
      val = min_valuing(0, gameState.generateSuccessor(0, legalAction), 1)
      if val > max_val:
        max_val = val
        acting = legalAction
    return acting


    # END_YOUR_ANSWER

######################################################################################
# Problem 2b: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER (our solution is 42 lines of code, but don't worry if you deviate from this)
    # for pacman agent who should maximize
    def max_valuing(depth, state, a, b):
      if self.depth == depth or state.isLose() or state.isWin():
        return self.evaluationFunction(state)
      
      alpha = a
      val = float('-inf')
      legalActions = state.getLegalActions(0)
      for legalAction in legalActions:
        val = max(val, min_valuing(depth, state.generateSuccessor(0, legalAction), 1, alpha, b))
        if val > b:   # pruning step
          return val
        alpha = max(alpha, val)
      return val

    # for ghost agent who should minimize
    def min_valuing(depth, state, agentIndex, a, b):
      if self.depth == depth or state.isLose() or state.isWin():
        return self.evaluationFunction(state)
      
      beta = b
      val = float("inf")
      legalActions = state.getLegalActions(agentIndex)

      #checking whether this is final ghost or not
      if agentIndex == state.getNumAgents()-1:
        for legalAction in legalActions:
          val = min(val, max_valuing(depth+1, state.generateSuccessor(agentIndex, legalAction), a, beta))
          if val < a:
            return val
          beta = min(val, beta)
      else:
        for legalAction in legalActions:
          val = min(val, min_valuing(depth, state.generateSuccessor(agentIndex, legalAction), agentIndex + 1, a, beta))
          if val < a:
            return val
          beta = min(val, beta)
      return val        

    #moving pacman : find the acting which makes maximum value of successors' value 
    # initialize a and b, val
    max_val = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    acting = Directions.STOP

    legalActions = gameState.getLegalActions(0)
    for legalAction in legalActions:
      val = min_valuing(0, gameState.generateSuccessor(0, legalAction), 1, alpha, beta)
      if val > max_val:
        max_val = val
        acting = legalAction
      if val > beta:
        return acting
      alpha = max(val, alpha)
    return acting

    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)

    #for pacman agent who should choose maximize
    def max_valuing(depth, state):
      if self.depth == depth or state.isLose() or state.isWin():
        return self.evaluationFunction(state)

      val = float('-inf')
      legalActions = state.getLegalActions(0)
      for legalAction in legalActions:
        val = max(val, expect_valuing(depth, state.generateSuccessor(0, legalAction), 1))
      return val

    # for ghost agent who choose action randomly
    def expect_valuing(depth, state, agentIndex):
      if self.depth == depth or state.isLose() or state.isWin():
        return self.evaluationFunction(state)

      tot_val = 0
      legalActions = state.getLegalActions(agentIndex)
      Acting_num = len(legalActions)
      # checking whether the final ghost's turn or not
      if agentIndex == state.getNumAgents()-1:
        for legalAction in legalActions:
          tot_val += max_valuing(depth + 1, state.generateSuccessor(agentIndex, legalAction))
      else:
        for legalAction in legalActions:
          tot_val += expect_valuing(depth, state.generateSuccessor(agentIndex, legalAction), agentIndex + 1)
      # uniformly random choice -> calculate the average
      average_val = tot_val/Acting_num
      return average_val


    # moving pacman
    max_val = float('-inf')
    acting = Directions.STOP

    legalActions = gameState.getLegalActions(0)
    for legalAction in legalActions:
      val = expect_valuing(0, gameState.generateSuccessor(0, legalAction), 1)
      if val > max_val:
        max_val = val
        acting = legalAction
    return acting


    # END_YOUR_ANSWER

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 4).
  """

  # BEGIN_YOUR_ANSWER (our solution is 60 lines of code, but don't worry if you deviate from this)
  # For evaluation, we should consider 3 aspects
  # 1. living (because when pacman dies, 500 decrease => enormous damage)
  # 2. eating capsules and eat ghosts   (eating ghost +200)
  # 3. eating foods


  utility = currentGameState.getScore()

  pac_pos = currentGameState.getPacmanPosition()
  ghosts = currentGameState.getGhostStates()

  foods = currentGameState.getFood().asList()  
  food_num = currentGameState.getNumFood()
  if food_num > 0:
    sum_food_distances = 0
    foods_distances = [manhattanDistance(food, pac_pos) for food in foods]
    sum_food_distances = sum(foods_distances+[1])
    utility += 100 / (food_num + 1) + 100 / (sum_food_distances+1)

  
  sum_scaredtime = 0

  # how to deal with Ghosts
  for ghost in ghosts:
    sum_scaredtime += ghost.scaredTimer
    ghost_distance = manhattanDistance(pac_pos, ghost.getPosition())
    if ghost_distance < 3:
      if ghost.scaredTimer > 1:    # eating ghost not nearby ghost respawn
       utility += 400 / (ghost_distance + 1)
       if pac_pos == ghost.getPosition():
         utility += 500
      else:                       # run away from ghosts
        utility -= 700 / (ghost_distance + 1)
    else:
      if ghost.scaredTimer > 1:
        utility += 400 / (ghost_distance + 1)

  capsules = currentGameState.getCapsules()
  cap_num = len(capsules)

  if cap_num > 0:
    cap_distances = [manhattanDistance(cap, pac_pos) for cap in capsules]
    closest_cap_distance = min(cap_distances)
    if sum_scaredtime > 0:
      if closest_cap_distance < 3:
        utility -= 300 / (closest_cap_distance + 1)
    else:
      utility += 300 / (closest_cap_distance + 1)
      # pacman won't eat capsule unless eating capsule gives it reward.
      if sum_scaredtime > 39 * (currentGameState.getNumAgents()-1):
        utility += 500
  

  return utility

  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
