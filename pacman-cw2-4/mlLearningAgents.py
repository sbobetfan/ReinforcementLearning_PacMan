# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py


# **********************************************************************
#                   THIS CODE HAS BEEN ADAPTED FOR THE
#                  MACHINE LEARNING 6CCS3ML1 COURSEWORK 2
#
# Martin Johnston, MSc Advanced Computing
# Student no. 20097129
#
# The following code has been completed to implement the Q-Learning algorithm.
#
# **********************************************************************

from pacman import Directions
from game import Agent
import random
import game
import util
import math

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Local copy of the game score
        self.scoreTracker = 0

        # Keep a record of the actions taken and states visited previously.
        # We need the previous Q Values in order to calculate the updated
        # Q Values, which is where this record will be useful.
        self.PreviousStates = []
        self.PreviousActions = []

        # Initialise record of these Q-Values
        self.q_value = util.Counter()


    # Function to reset the values of the current episode,
    # to prepare for the next one
    def RESET_VALUES(self):
        self.scoreTracker = 0
        self.PreviousStates = []
        self.PreviousActions = []

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def getEpsilon(self):
        return self.epsilon

    def setEpsilon(self, value):
        self.epsilon = float(value)

    # The value of epsilon is allowed to decrease
    # as more training episodes elapse.
    # This makes PacMan greedier as it learns more.
    def getNewEpsilon(self):
        TrainingEpisodesSoFar = self.getEpisodesSoFar()
        TotalTrainingEpisodes = self.getNumTraining()
        epsilon = 0.1*(1 - (TrainingEpisodesSoFar*1.0/TotalTrainingEpisodes))
        return epsilon

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = float(value)

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # Accessor functions for the previous action and state
    def getPreviousState(self):
        return self.PreviousStates[-1]

    def getPreviousAction(self):
        return self.PreviousActions[-1]

    # Getter and setter methods for the Q Value of a given state/action pair
    def getQValue(self, state, action):
        return self.q_value[(state,action)]

    def updateQValue(self, state, action, reward, maxQ):
        current_QValue= self.getQValue(state,action)
        alpha = self.getAlpha()
        gamma = self.getGamma()

        # Update rule for Q-Learning
        updated_QValue = current_QValue + alpha*(reward + gamma*maxQ - current_QValue)

        self.q_value[(state,action)] = updated_QValue

    # return the max Q Value of a state
    def getMaxQ(self, state):
        LegalActions = state.getLegalPacmanActions()
        # For each legal action, build a list with the current q-values:
        Q_Values = []
        for action in LegalActions:
            QVal = self.getQValue(state,action)
            Q_Values.append(QVal)
        # if there are no legal actions, return 0
        if len(Q_Values) == 0:
            return 0
        # Return the maximum Q Value
        return max(Q_Values)

    # Accessor function for the reward of an action.
    # Reward is calculated as the change in score due to the last action.
    def getReward(self, state):
        reward = state.getScore()-self.scoreTracker
        return reward

    # Accessor function for the reverse of a given action.
    def getReverseOfAction(self, action):
        return Directions.REVERSE[action]

    # Method to calculate the Euclidean distance between two points.
    def Euclid_dist(self, point1, point2):
        x_distance = point1[0] - point2[0]
        y_distance = point1[1] - point2[1]
        return math.sqrt(x_distance**2 + y_distance**2)


    # ACKNOWLEDGEMENT: The method below takes inspiration from the method doTheRightThing found from this source: https://github.com/Lamicc/Pacman_Qlearning/blob/3828e39b51c2c5d77bd7515c98eae14f6be2d021/mlLearningAgents.py
    # This method has not been directly copied and pasted; adaptations have been made with further documentation to illustrate my understanding of this method's intuition.
    #
    def calculatedMove(self, state):
        legal = state.getLegalPacmanActions()

        # To force Exploration, the first half of the training session forces
        # PacMan not to stop or turn back - so long as the ghost is not nearby.
        TrainingEpisodesSoFar = self.getEpisodesSoFar()
        TotalTrainingEpisodes = self.getNumTraining()
        # Limit this exploration to the first half of training:
        if (TrainingEpisodesSoFar*1.0 / TotalTrainingEpisodes) < 0.5:
            # Dont stop:
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)

            # Dont turn back if it is not necessary:
            if len(self.PreviousActions) > 0:

                # Retrieve x and y positions of PacMan and the Ghost
                PacMan_position = (state.getPacmanPosition()[0], state.getPacmanPosition()[1])
                Ghost_position = (state.getGhostPosition(1)[0], state.getGhostPosition(1)[1])

                # Retrieve the previous action taken by PacMan
                Previous_Action = self.getPreviousAction()
                ReverseOfPrevAction = self.getReverseOfAction(Previous_Action)

                # calculate the Euclidean distance between the ghost and pacman
                PacMan_Ghost_distance = self.Euclid_dist(PacMan_position, Ghost_position)

                # If far away from the ghost, and moves are still available,
                # do not turn back.
                if PacMan_Ghost_distance > 2:
                    if (ReverseOfPrevAction in legal) and len(legal)>1:
                        legal.remove(ReverseOfPrevAction)

        # For each legal action, build a list with the corresponding current Q Values:
        Q_Vals = util.Counter()
        for action in legal:
          Q_Vals[action] = self.getQValue(state, action)
        # Return the action with maximum Q Value
        return Q_Vals.argMax()

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # print "Legal moves: ", legal
        # print "Pacman position: ", state.getPacmanPosition()
        # print "Ghost positions:" , state.getGhostPositions()
        # print "Food locations: "
        # print state.getFood()
        # print "Score: ", state.getScore()

        # recalculate Q Value
        # Reward is calculated as the change in score due to the last action
        reward = self.getReward(state)
        if len(self.PreviousStates) > 0:
            # Retrieve the most recent action and state
            Previous_State = self.getPreviousState()
            Previous_Action = self.getPreviousAction()
            # Retrieve the maximum Q value of all state/value pairs
            # for this state
            maxQ = self.getMaxQ(state)
            # Now calculate and update the new Q value
            self.updateQValue(Previous_State, Previous_Action, reward, maxQ)


        # Make the epsilon-greedy choice
        choice = random.random()
        if choice <= (1 - self.getEpsilon()):
            pick = self.calculatedMove(state)
        else:
            pick =  random.choice(legal)


        # update score and record of previous states and actions
        self.scoreTracker = state.getScore()
        self.PreviousStates.append(state)
        self.PreviousActions.append(pick)

        return pick

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        # recalculate Q Values
        # Reward is calculated as the change in score due to the last action
        reward = self.getReward(state)
        # Retrieve the most recent action and state
        Previous_State = self.getPreviousState()
        Previous_Action = self.getPreviousAction()
        self.updateQValue(Previous_State, Previous_Action, reward, 0)

        # End of episode - reset score and State/Action records for next episode
        self.RESET_VALUES()

        # decrease epsilon during the training
        # This makes PacMan greedier as it learns more
        NewEpsilon = self.getNewEpsilon()
        self.setEpsilon(NewEpsilon)

        # print "A game just ended!"

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() % 100 == 0:
            print "Completed %s training episodes" % self.getEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
