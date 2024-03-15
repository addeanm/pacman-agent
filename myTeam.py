# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

import random
import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from time import time

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

class OffensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.weights = {'flag_score': 100, 'distance_to_flag': -1, 'distance_to_food': -1, 'nearby_enemy': -100}  # Updated weights
        self.alpha = 0.1  # Learning rate
        self.enemy_nearby = False
        self.eat_enemy_timer = None

    def update_weights(self, features, reward):
        # Update weights based on Q-learning
        for feature in features:
            self.weights[feature] += self.alpha * reward * features[feature]

    def choose_action(self, game_state):
        if len(self.get_food(game_state).as_list()) <= 4 and game_state.is_red:
            # If four or fewer food pellets remaining on our side, focus on catching enemy agents
            actions = game_state.get_legal_actions(self.index)
            for action in actions:
                successor = self.get_successor(game_state, action)
                my_pos = successor.get_agent_position(self.index)
                enemy_positions = [successor.get_agent_position(i) for i in self.get_opponents(successor)]
                min_enemy_distance = min(self.get_maze_distance(my_pos, enemy_pos) for enemy_pos in enemy_positions)
                if min_enemy_distance <= 5:
                    self.enemy_nearby = True
                    return action
            self.enemy_nearby = False

        return super().choose_action(game_state)

    def evasive_action(self, game_state, best_actions):
        # Choose evasive action (opposite direction to the nearest enemy Pacman)
        enemy_positions = [game_state.get_agent_position(i) for i in self.get_opponents(game_state)]
        nearest_enemy_dist = min(self.get_maze_distance(game_state.get_agent_position(self.index), enemy_pos, game_state)
                                 for enemy_pos in enemy_positions)
        nearest_enemy_pos = next(pos for pos in enemy_positions
                                 if self.get_maze_distance(game_state.get_agent_position(self.index), pos, game_state) == nearest_enemy_dist)
        directions_to_enemy = game_state.get_directions(game_state.get_agent_position(self.index), nearest_enemy_pos)
        opposite_direction = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        evasive_actions = [action for action in best_actions if action != opposite_direction]
        if evasive_actions:
            return random.choice(evasive_actions)
        else:
            return random.choice(best_actions)

    def is_safe_action(self, game_state, action):
        # Check if the action results in moving closer to any enemy agents
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_position(self.index)
        enemy_positions = [successor.get_agent_position(i) for i in self.get_opponents(successor)]
        for enemy_pos in enemy_positions:
            if self.get_maze_distance(my_pos, enemy_pos, game_state) <= 5:
                return False
        return True

    def get_features(self, game_state, action):
        features = util.Counter()
        my_pos = game_state.get_agent_position(self.index)
        if game_state.is_red and self.index % 2 == 0:
            # If on own side and an even index agent (a ghost), prioritize reaching the opponent's flag
            closest_flag = min(game_state.get_blue_team_indices(),
                               key=lambda i: self.get_maze_distance(my_pos, game_state.get_agent_position(i)))
            features['flag_score'] = -1
            features['distance_to_flag'] = self.get_maze_distance(my_pos, game_state.get_agent_position(closest_flag), game_state)
        else:
            # Focus on eating food pellets while avoiding getting too close to enemy agents
            food_positions = [food for food in self.get_food(game_state).as_list() if food[0] > game_state.data.layout.width / 2]
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, food, game_state) for food in food_positions)

            # Check if enemy Pacman is nearby
            enemy_positions = [game_state.get_agent_position(i) for i in self.get_opponents(game_state)]
            min_enemy_distance = min(self.get_maze_distance(my_pos, enemy_pos, game_state) for enemy_pos in enemy_positions)
            if min_enemy_distance <= 5:
                self.enemy_nearby = True
                features['nearby_enemy'] = 1
            else:
                self.enemy_nearby = False
                features['nearby_enemy'] = 0

        return features

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        return sum(features[feature] * self.weights[feature] for feature in features)

    def observe_transition(self, game_state, action, next_game_state, reward):
        # Q-learning update
        features = self.get_features(game_state, action)
        self.update_weights(features, reward)

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.eat_enemy_timer = None

    def choose_action(self, game_state):
        if len(self.get_food(game_state).as_list()) <= 4 and game_state.is_red:
            # If four or fewer food pellets remaining on our side, focus on catching enemy agents
            actions = game_state.get_legal_actions(self.index)
            for action in actions:
                successor = self.get_successor(game_state, action)
                my_pos = successor.get_agent_position(self.index)
                enemy_positions = [successor.get_agent_position(i) for i in self.get_opponents(successor)]
                min_enemy_distance = min(self.get_maze_distance(my_pos, enemy_pos, game_state) for enemy_pos in enemy_positions)
                if min_enemy_distance <= 5:
                    self.enemy_nearby = True
                    return action
            self.enemy_nearby = False

        return super().choose_action(game_state)



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
