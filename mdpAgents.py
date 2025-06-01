# !/usr/bin/env python
# -*- coding: utf-8 -*-

# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import math
import matplotlib.pyplot as plt
import numpy as np

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print "Starting up MDPAgent!"
        name = "Pacman"
        self.states = []
        self.gamma = .9
        self.rewards = {}
        self.outer_ring = []
        self.number_of_live_ghosts = 0
        self.spawn_positions = [(8,5),(9,5),(10,5),(11,5),(10,6),(11,6)]
        self.negative_positions = {}
        self.smallGrid = False
        self.mediumClassic = False
        self.environment = None

    def compute_all_positions(self, state):
        corners = api.corners(state)
        bottom_left = (min(x for x, y in corners), min(y for x, y in corners))
        x_min = bottom_left[0]
        y_min = bottom_left[1]
        top_right = (max(x for x, y in corners), max(y for x, y in corners))
        x_max = top_right[0]
        y_max = top_right[1]
        if x_max == 6:
            self.smallGrid = True
        else:
            self.mediumClassic = True
        return [(x,y) for x in range(x_min, x_max+1) for y in range(y_min, y_max+1)]

    def define_outer_ring(self, x_min, x_max, y_min, y_max):
        for j in range(y_min+1, y_max):
            # to take us from 1 (just above the bottom wall)
            # to just below the top wall
            if (x_min+1, j) in self.states:
                self.outer_ring.append((x_min+1, j))
            if (x_max-1, j) in self.states:
                self.outer_ring.append((x_max-1, j))
        for i in range(x_min+2, x_max-1):
            if (i, y_min+1) in self.states:
                self.outer_ring.append((i, y_min+1))
            if (i, y_max-1) in self.states:
                self.outer_ring.append((i, y_max-1))

    def visualize_utilities(self, utilities, x_max, y_max):
        grid = np.full((y_max + 1, x_max + 1), np.nan)
        for (x, y), value in utilities.items():
            grid[y, x] = value  # Note the order: [y, x]
        plt.imshow(grid, cmap='gray', origin='lower')
        for (x, y), value in utilities.items():
            if (x,y) == self.pacman_location:
                plt.text(x, y, '*', ha='center', va='center', color='blue', fontsize=12, fontweight='bold')
            plt.text(x, y, "{:.0f}".format(value), ha="center", va="center", color="red")
        plt.colorbar()
        plt.draw()
        plt.pause(.01)
        plt.clf()

    def north(self, utilities, position):
        # where position is a tuple of coords
        # check north, east, west
        current_utility = utilities[position]
        x, y = position

        position_above = (x, y+1)
        position_left = (x-1, y)
        position_right = (x+1, y)

        utility_above = utilities.get(position_above, current_utility)
        utility_left = utilities.get(position_left, current_utility)
        utility_right = utilities.get(position_right, current_utility)

        return (.8*utility_above + .1*utility_left +.1*utility_right)

    def south(self, utilities, position):
        # where position is a tuple of coords
        # check south, east, west
        current_utility = utilities[position]
        x, y = position

        position_below = (x, y-1)
        position_left = (x-1, y)
        position_right = (x+1, y)

        utility_below = utilities.get(position_below, current_utility)
        utility_left = utilities.get(position_left, current_utility)
        utility_right = utilities.get(position_right, current_utility)

        return (.8*utility_below + .1*utility_left +.1*utility_right)

    def west(self, utilities, position):
        # where position is a tuple of coords
        # check west, north, south
        current_utility = utilities[position]
        x, y = position

        position_left = (x-1, y)
        position_above = (x, y+1)
        position_below = (x, y-1)

        utility_below = utilities.get(position_below, current_utility)
        utility_left = utilities.get(position_left, current_utility)
        utility_above = utilities.get(position_above, current_utility)

        return (.8*utility_left + .1*utility_above +.1*utility_below)

    def east(self, utilities, position):
        # where position is a tuple of coords
        # check east, north, south
        current_utility = utilities[position]
        x, y = position

        position_right = (x+1, y)
        position_above = (x, y+1)
        position_below = (x, y-1)

        utility_below = utilities.get(position_below, current_utility)
        utility_right = utilities.get(position_right, current_utility)
        utility_above = utilities.get(position_above, current_utility)

        return (.8*utility_right + .1*utility_above +.1*utility_below)

    def utility_for_this_position(self, utilities, position):
        # we have the position for which we must calculate the new utility
        # we just need to check all 4 directions and calculate the expected utility as a result for each direction and then take a max
        # for a given direction (e.g. north) we know the position of the state that we need to look at- if that state isn't in self.states
        # then it must be a wall in which case pacman will remain in the same position
        return round(self.rewards[position] + self.gamma*max(round(self.north(utilities, position),2), round(self.south(utilities, position),2), round(self.west(utilities, position),2), round(self.east(utilities, position),2)),2)

    def one_step_of_val_iter(self, utilities):
        # this literally takes in the utilities and tells you what the utilities will be as a result, given gamma and R(s) of course
        # in a sense most of the work happens here
        return {position: self.utility_for_this_position(utilities, position) for position in utilities}

    def calculate_distance_to_nearest_ghost(self, ghosts, position):
        # go through the ghosts list and take the ghost that is nearest by euclidean dist
        distance_to_ghosts = []
        for ghost in ghosts:
            x_1, y_1 = ghost
            x_2, y_2 = position
            distance_to_ghosts.append((x_1-x_2)**2 + (y_1-y_2)**2)
        # return dist
        return math.sqrt(min(distance_to_ghosts))

    def no_of_legal_moves(self, position):
        number_of_legal_moves = 0
        x_pos, y_pos = position
        for direction in [(-1,0), (1,0), (0,1), (0,-1)]:
            x, y = direction
            # e.g. pos (4,5) with dir (1,0) results in (5,5)
            if (x_pos+x, y_pos+y) in self.states:
                number_of_legal_moves += 1
        return number_of_legal_moves

    def get_legal_moves(self, position):
        x, y = position
        moves = []
        for dx, dy in [(-1,0), (1,0), (0,1), (0,-1)]:
            next_pos = (x + dx, y + dy)
            if next_pos in self.states:
                moves.append((dx, dy))
        return moves

    def return_direction_of_ghost_if_no_fork(self, original_direction, position):
        # we will have already checked that there are only 2 legal moves at this position
        # so no need for that check here
        x_pos, y_pos = position
        directions = [(-1,0), (1,0), (0,1), (0,-1)]
        x, y = original_direction
        # e.g. if original direction was (1,0) then we don't want to consider
        # the direction (-1, 0) because that looks at the positions back where we came from
        # we want the other direction (whatever that may be) that leads us out and through
        direction_we_do_not_want = (-x, -y)
        directions.remove(direction_we_do_not_want) # assuming not (0,0)
        for direction in directions:
            x, y = direction
            # this is what we're looking at: a position where there are only 2 legal moves
            # we know that the ghost will just continue through given we know which way it's coming from
            # i.e. from south or east
            #    ________
            #    | * * *    
            #    | * |
            #    | * |
            # e.g. pos (4,5) with dir (1,0) results in (5,5)
            if (x_pos+x, y_pos+y) in self.states:
                return direction # this is that direction that leads through the passage
    
    def return_directions_here(self, position, direction):
        # we don't include the way we came
        # NEVER ALLOW (0,0) to be passed to this
        directions = [(1,0), (-1, 0), (0, 1), (0, -1)]
        x_pos, y_pos = position
        x, y = direction
        direction_we_came_from = (-x, -y)
        directions.remove(direction_we_came_from)
        for possibility in directions:
            i, j = possibility
            # now check the 2 directions to see if we run into a wall- if so, remove from list
            # so if we hit into walls for both directions there will be nothing left in the list- 2 legal moves here then
            if (x_pos+i, y_pos+j) not in self.states:
                directions.remove(possibility)
        return directions

    def append_to_dict(self, dictionary, key, value):
        # Appends a value to a dictionary entry. If the key exists, adds the value to the existing value.
        if key in dictionary:
            dictionary[key] += value
        else:
            dictionary[key] = value
        return dictionary

    def strength_of_signal(self, distance):
        # this is so that we take the distance of the position from the ghost
        # along a given path
        # and reduce the strength if it's further away because we're less afraid
        return 1000-50*distance
    
    # TODO: comment
    def ghost_line_of_sight_recursive(self, direction, position, p, distance_from_ghost):
        # distance_from_ghost tells you how far down you are this exploratory path of a possible
        # paths that the ghost might take- we choose to explore only 3 steps in advance
        if distance_from_ghost < 3 and direction is not None:
            x_pos, y_pos = position
            x, y = direction
            # for this direction of choice, calculate the next position (i.e. the next state down this particular path)
            # - this will be another addition to self.negative_positions as it is a position that the ghost can achieve in 3 moves or less
            next_position = (x_pos + x, y_pos + y)
            distance_from_ghost += 1
            if next_position in self.states:
                num_legal_moves = self.no_of_legal_moves(next_position)
                self.negative_positions = self.append_to_dict(self.negative_positions, next_position, p * self.strength_of_signal(distance_from_ghost))
                if num_legal_moves == 2:

                    #                      | * |                                                                   _________
                    # e.g                  | * |      next position calculated as a result ──────────────────────>| * ______
                    #     ghost denoted    | * |      has only 2 legal moves- here on the left (0,1)              | * |<─────────┤
                    #     by G             | * |      further up the path or (0,-1)                               | * |          |
                    #                      | * |      which is back down the path towards the ghost               | G |          |
                    #                      | * |<──────┤ we then add this position with the associated            | * |          current
                    #                 ├────| * |         value determined by:                                     | * |          position
                    #                 |    | * |           (probability of arriving here)*(f(distance_to_ghost))  | * |
                    #   we're here────┤    | G | chosen dir = (0, 1)    where f is a linearly decreasing function | * |          
                    #                      | * |<─────────── either no turn ──────or────there is a turn──────────>| * |
                    #    (position of      | * |             as on the left             but the logic             | * |
                    #     ghost we're      | * |                (no fork)               does not change           | * |
                    #     considering)     | * |                                        we just use a function    | * |
                    #                      | * |                                        to determine the dir      | * |
                    #                      | * |                                        from next_position        | * |
                    #                      | * |                                        (at the turn)             | * |

                    new_direction = self.return_direction_of_ghost_if_no_fork(direction, next_position)
                    self.ghost_line_of_sight_recursive(new_direction, next_position, p, distance_from_ghost)
                
                #                                                                                4 legal moves ──────\  | * |
                #                                                                                                     \ | ? |                                                                      
                #                                       for each direction,                                            ┘| ^ |
                #                                       one calls the function                                    ──────┤ | ├──────
                #                                       again with the distance                                   * ? < ─ ↑ ─ > ?   
                #                                       incremented and probability                               ──────┤ | ├────── 
                #                  3 legal moves        reduced so that the algorithm                        reduced    | | |
                #                        ↓              can continue exploring                             probability  | G |         
                #                ─────────────────      and putting more states into                        for each    | * |
                #                * * * * * * * * *      self.negative_positions                             direction   | * |
                #                ──────┤ * ├──────      with the appropriate values                            at       | * |
                #                      | * |                i.e. negativity                                 junction    | * |
                #                 ├────| * |                                                                            | * |
                #                 |    | * |                                                                            | * |
                #   we're here────┤    | G |                                                                            | * |          
                #                      | * |<─────────── either 3 legal moves ────────or────────4 legal moves──────────>| * |
                #    (position of      | * |             at fork as on the left                (essentially the
                #                                                                               same thing)    
                
                elif num_legal_moves >= 3:
                    for way_out in self.return_directions_here(next_position, direction):
                        self.ghost_line_of_sight_recursive(way_out, next_position, 0.8 * p, distance_from_ghost)

    def recursive_tree_of_positions(self, ghost_position, direction):
        # we need a dictionary that associates states with values- namely, what we want in the end is
        # the sum of (probability of arriving here)*(f(distance)) where f is some decreasing function because it's more safe
        # what we'll do is for each new position that we come across we'll add it on with the given probability*f(distance)
        # if the position is already there and we come across it once more then we'll add the probability*f(distance) that we currently have

        # we can just use the current direction of the ghost to determine whether or not the given position that we consider as
        # we go along has more than 2 legal moves
        # i.e. remove the direction from the list [(-1,0), (1,0), (0,1), (1,0)] and then check how many of the resulting directions run into walls
        # or valid positions using self.states- if there are 2 such directions this is a 3-way fork (including direction we came from)
        # , 3 then a 4-way fork (including direction we came from)
        x_pos, y_pos = ghost_position
        x, y = direction
        next_position = (x_pos + x, y_pos + y)
        if next_position in self.states:
            self.ghost_line_of_sight_recursive(direction, ghost_position, 1, 0)
    
    def recursive_tree_of_positions_small_grid(self, ghost_position, direction):
        # we need a dictionary that associates states with values- namely, what we want in the end is
        # the sum of (probability of arriving here)*(f(distance)) where f is some decreasing function because it's more safe
        # what we'll do is for each new position that we come across we'll add it on with the given probability*f(distance)
        # if the position is already there and we come across it once more then we'll add the probability*f(distance) that we currently have

        # we can just use the current direction of the ghost to determine whether or not the given position that we consider as
        # we go along has more than 2 legal moves
        # i.e. remove the direction from the list [(-1,0), (1,0), (0,1), (1,0)] and then check how many of the resulting directions run into walls
        # or valid positions using self.states- if there are 2 such directions this is a 3-way fork (including direction we came from)
        # , 3 then a 4-way fork (including direction we came from)
        x_pos, y_pos = ghost_position
        x, y = direction
        next_position = (x_pos + x, y_pos + y)
        if next_position in self.states:
            self.ghost_line_of_sight_recursive_small_grid(direction, ghost_position, 1, 0)
    
    def ghost_line_of_sight_recursive_small_grid(self, direction, position, p, distance_from_ghost):
        if distance_from_ghost < 6 and direction is not None:
            x_pos, y_pos = position
            x, y = direction
            next_position = (x_pos + x, y_pos + y)
            distance_from_ghost += 1
            if next_position in self.states:
                num_legal_moves = self.no_of_legal_moves(next_position)
                self.negative_positions = self.append_to_dict(self.negative_positions, next_position, p * self.strength_of_signal(distance_from_ghost))
                if num_legal_moves == 2:
                    new_direction = self.return_direction_of_ghost_if_no_fork(direction, next_position)
                    self.ghost_line_of_sight_recursive_small_grid(new_direction, next_position, p, distance_from_ghost)
                elif num_legal_moves >= 3:
                    for way_out in self.return_directions_here(next_position, direction):
                        self.ghost_line_of_sight_recursive_small_grid(way_out, next_position, float(p)/(self.no_of_legal_moves(next_position)-1), distance_from_ghost)
    
    def food_scaling_factor(self, amount_of_food):
        if amount_of_food < 40:
            return math.pow(1.05, (40-amount_of_food))
        else:
            return 1

    def compute_capsule_reward(self, capsule_position, alive_ghosts):
        # WE ONLY EVER USE THIS IF BOTH GHOSTS ARE LIVE (NOT SCARED)
        # Compute distances from the capsule to each alive ghost
        distances = [self.calculate_distance_to_nearest_ghost([ghost], capsule_position) for ghost in alive_ghosts]
        # Get the minimum distance
        use_adjusted_reward = True
        for distance in distances:
            if distance > 5:
                use_adjusted_reward = False
        # set it so that we only use the adjusted reward if the distances of both live ghosts is less than 10- i.e both ghosts are close
        
        if use_adjusted_reward:
            min_distance = min(distances)
            # Adjust the reward based on the minimum distance
            # Using an exponential decay function for the reward
            base_capsule_reward = -50
            ghost_reward_scale = 500  # Adjust this scaling factor as needed
            k = 0.5  # Decay rate; adjust for sensitivity
            adjusted_reward = base_capsule_reward + ghost_reward_scale * math.exp(-k * min_distance)
        else:
            return -50
        return adjusted_reward

    def compute_value_based_on_condition_ghosts_far(self, env, state):
        self.negative_positions = {}
        ghost_positions = [(int(math.ceil(x)), int(math.ceil(y))) for (x, y) in api.ghosts(env)]
        blue_ghost = ghost_positions[0]
        red_ghost = ghost_positions[1]
        ghosts_states = api.ghostStatesWithTimes(env)
        ghost_position_to_scaredTimer= {(int(math.ceil(x)), int(math.ceil(y))): scared_timer for (x, y), scared_timer in ghosts_states}
        alive_ghosts = [coord for coord in ghost_position_to_scaredTimer if ghost_position_to_scaredTimer[coord]==0] # positions of non-scared ghosts
        ghost_attack_positions = []
        amount_of_food = len(api.food(env)) # how much food is on the map- once less than 10 we'll up the reward for the last food remaining
        # positions_of_outer_ring_with_food = [position for position in self.outer_ring if position in api.food(env)]
        T = 40 # number of moves for which ghost will be scared once it gets scared
        reward = 0
        if state in self.spawn_positions:
            reward -= 1000
        for ghost in ghost_positions:
            if ghost in alive_ghosts:
                # i.e. alive so we should predict where it's going
                # and set self.negative_positions
                for direction in self.get_legal_moves(state):
                    self.recursive_tree_of_positions(ghost, direction)
        
        if state in ghost_positions:
                reward -= 100
        if state in api.capsules(env):
            # a capsule
            # if state in positions_of_outer_ring_with_food:
            #     reward += 50
            # reward -= 100
            # if len(api.food(env)) == 1:
            #     reward += 200
            reward += 50
        elif state in api.food(env):
            # capsules get diff reward to food hence why an elif here as it's a diff case
            # if state in positions_of_outer_ring_with_food:
            #     reward += (self.food_scaling_factor(amount_of_food)*35)
            reward += (self.food_scaling_factor(amount_of_food)*35)
        else:
            # this will be the case where the square is empty- no ghosts, no capsules, no food
            if len(api.food(env)) == 1 and len(api.capsules(env)) != 0:
                # Penalize consuming the last food pellet before capsules are used
                reward -= 500
            reward -= 1
        return reward


    def compute_value_based_on_condition(self, env, state):
        self.negative_positions = {}
        self.positive_positions = {}
        ghost_positions = [(int(math.ceil(x)), int(math.ceil(y))) for (x, y) in api.ghosts(env)]
        blue_ghost = ghost_positions[0]
        red_ghost = ghost_positions[1]
        ghosts_states = api.ghostStatesWithTimes(env)
        ghost_position_to_scaredTimer= {(int(math.ceil(x)), int(math.ceil(y))): scared_timer for (x, y), scared_timer in ghosts_states}
        alive_ghosts = [coord for coord in ghost_position_to_scaredTimer if ghost_position_to_scaredTimer[coord]==0] # positions of non-scared ghosts
        ghost_attack_positions = []
        amount_of_food = len(api.food(env)) # how much food is on the map- once less than 10 we'll up the reward for the last food remaining
        T = 40 # number of moves for which ghost will be scared once it gets scared
        reward = 0
        if state in self.spawn_positions:
            reward -= 1000
        for ghost in ghost_positions:
            if ghost in alive_ghosts:
                # i.e. alive so we should predict where it's going
                # and set self.negative_positions
                legal_moves = self.get_legal_moves(ghost)
                for direction in legal_moves:
                    self.recursive_tree_of_positions(ghost, direction)

        if state in ghost_positions:
            if ghost_position_to_scaredTimer[state] > 0:
                # ghost is scared
                if len(alive_ghosts) > 0:
                    # throw alive_ghosts into here because we want to know how far the scared ghost is from the live ghost
                    # we don't want to run headlong into a scenario where a scared ghost is near a live one
                    # also cap the multiplier by 1- we don't want to artificially raise the value of the scared ghost
                    # just because the live one is far off
                    bounded_distance = min(self.calculate_distance_to_nearest_ghost(alive_ghosts, state) / 4, 1)
                    reward += bounded_distance*(2000*(ghost_position_to_scaredTimer[state]/float(T)))
                else:
                    # no ghosts live, all scared
                    reward += (2000*(ghost_position_to_scaredTimer[state]/float(T)))
            else:
                # ghost is live, not scared
                reward -= 1000
        if state in api.capsules(env):
            # a capsule
            if state in self.negative_positions:
                reward -= self.negative_positions[state]
            if len(alive_ghosts) != 0:
                if self.calculate_distance_to_nearest_ghost(alive_ghosts, state) <= 1:
                    reward -= 1000
            # if len(alive_ghosts) == 2:
            #     adjusted_capsule_reward = self.compute_capsule_reward(state, alive_ghosts)
            #     reward += adjusted_capsule_reward
            # else:
            #     # no need to get another capsule if there's still a scared ghost on the map
            #     # we should be chasing the ghost not the capsule
            #     reward = -100
            # if len(api.food(env)) == 2:
            #     reward += 500
            reward += 50
        elif state in api.food(env):
            # capsules get diff reward to food hence why an elif here as it's a diff case
            if state in self.negative_positions:
                reward -= self.negative_positions[state]
            if len(alive_ghosts) != 0:
                if self.calculate_distance_to_nearest_ghost(alive_ghosts, state) <= 1:
                    reward -= 1000
            reward += self.food_scaling_factor(amount_of_food)*35
        else:
            # this will be the case where the square is empty- no ghosts, no capsules, no food
            if state in self.negative_positions:
                reward -= self.negative_positions[state]
            if len(alive_ghosts) != 0:
                if self.calculate_distance_to_nearest_ghost(alive_ghosts, state) <= 1:
                    reward -= 1000
            # if len(api.food(env)) == 1 and len(api.capsules(env)) != 0:
            #     # Penalize consuming the last food pellet before capsules are used
            #     reward -= 500
            reward -= 1
        return reward
    
    def compute_value_based_on_condition_small_grid(self, env, state):
        # reset negative positions
        self.negative_positions = {}

        # get ghost positions and scared timers
        ghost_positions = api.ghosts(env)

        reward = 0

        # get pacman's position
        pacman_position = api.whereAmI(env)
        pacman_x, pacman_y = pacman_position

        # get the list of food positions
        food_positions = api.food(env)
        num_food_left = len(food_positions)

        # assuming only one ghost on the small grid
        ghost_position = ghost_positions[0]
        ghost_x, ghost_y = ghost_position

        # calculate the mirrored position relative to the ghost
        # i.e. we want to incentivise pacman to go to the opposite side of the grid
        # as this where he is most likely to be safe as he puts the most distance between himself and the ghost
        x_max, y_max = 6, 6  # using grid size for small grid
        mirrored_x = x_max - ghost_x
        mirrored_y = y_max - ghost_y
        mirrored_position = (mirrored_x, mirrored_y)
        if (ghost_x, ghost_y) == (3,3):
            mirrored_position = (ghost_x-2, ghost_y)

        # up the reward for the opposite position to the ghost
        if state == mirrored_position and num_food_left == 2:
            reward += 1000
        elif state == mirrored_position:
            reward += 100

        ghost_on_left = ghost_x < 2 or (ghost_y < 2 and ghost_x < 3)

        if state in food_positions:
            if state == (3,3):
                if num_food_left == 1:
                    reward += 250
                elif ghost_on_left:
                    reward += 2000
                else:
                    reward -= 50
            else:
                # food, but not in the middle
                reward += 50
        elif state in ghost_positions:
            reward -= 100
        
        if self.calculate_distance_to_nearest_ghost(ghost_positions, state) <= 1:
            reward -= 100
        # if self.calculate_distance_to_nearest_ghost([mirrored_position], state) <= 1:
        #     reward += 1000

        legal_moves = self.get_legal_moves(ghost_positions[0])
        for direction in legal_moves:
            self.recursive_tree_of_positions_small_grid(ghost_positions[0], direction)
        if state in self.negative_positions:
            reward -= self.negative_positions[state]
        return reward

    def convert_env_to_utilities_dict(self, env):
        ghost_positions = [(int(math.ceil(x)), int(math.ceil(y))) for (x, y) in api.ghosts(env)]
        ghosts_states = api.ghostStatesWithTimes(env)
        ghost_position_to_scaredTimer= {(int(math.ceil(x)), int(math.ceil(y))): scared_timer for (x, y), scared_timer in ghosts_states}
        alive_ghosts = [coord for coord in ghost_position_to_scaredTimer if ghost_position_to_scaredTimer[coord]==0] # positions of non-scared ghosts
        if self.mediumClassic:
            if len(alive_ghosts) == 2:
                if self.calculate_distance_to_nearest_ghost(alive_ghosts, self.pacman_location) > 5:
                    self.rewards = {
                        state: self.compute_value_based_on_condition_ghosts_far(env, state) for state in self.states
                    }
                else:
                    self.rewards = {
                        state: self.compute_value_based_on_condition(env, state) for state in self.states
                    }
            else:
                self.rewards = {
                    state: self.compute_value_based_on_condition(env, state) for state in self.states
                }
        if self.smallGrid:
            self.rewards = {
                state: self.compute_value_based_on_condition_small_grid(env, state) for state in self.states
            }
        utilities = self.rewards.copy()
        return utilities

    def value_iteration(self, utilities, i):
        one_step = self.one_step_of_val_iter(utilities)
        while i < 50:
            # if i==0:
                # self.visualize_utilities(utilities, 20, 10)
            return self.value_iteration(one_step, i+1)
        # print("reached last iteration")
        # self.visualize_utilities(utilities, 20, 10)
        return one_step

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        self.environment = state
        self.states = list(filter(lambda pos: pos not in api.walls(state), self.compute_all_positions(state)))
        corners = api.corners(state)
        bottom_left = (min(x for x, y in corners), min(y for x, y in corners))
        x_min = bottom_left[0]
        y_min = bottom_left[1]
        top_right = (max(x for x, y in corners), max(y for x, y in corners))
        x_max = top_right[0]
        y_max = top_right[1]
        # self.define_outer_ring(x_min, x_max, y_min, y_max)
        ghosts_states = api.ghostStatesWithTimes(self.environment)
        ghost_position_to_scaredTimer= {(int(math.ceil(x)), int(math.ceil(y))): scared_timer for (x, y), scared_timer in ghosts_states}
        # sets the number of ghosts we're dealing with
        self.no_of_live_ghosts = len([coord for coord in ghost_position_to_scaredTimer if ghost_position_to_scaredTimer[coord]==0])
        print "Running registerInitialState for MDPAgent!"
        print "I'm at:"
        print api.whereAmI(state)
        
    # This is what gets run in between multiple games
    def final(self, state):
        print "Looks like the game just ended!"

    # For now I just move randomly
    def getAction(self, state):
        # Get the legal actions and remove 'STOP' if present
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        self.pacman_location = api.whereAmI(state)
        # Update the rewards based on the current state
        utilities = self.convert_env_to_utilities_dict(state)

        # Perform value iteration to compute the utilities
        utilities = self.value_iteration(utilities, 0)
        
        # Determine the best action from the current position
        current_position = api.whereAmI(state)
        action_utilities = {}
        for action in legal:
            if action == Directions.NORTH:
                expected_util = self.north(utilities, current_position)
            elif action == Directions.SOUTH:
                expected_util = self.south(utilities, current_position)
            elif action == Directions.EAST:
                expected_util = self.east(utilities, current_position)
            elif action == Directions.WEST:
                expected_util = self.west(utilities, current_position)
            action_utilities[action] = expected_util

        # Choose the action with the highest expected utility
        if action_utilities:
            # print(action_utilities)
            best_action = max(action_utilities, key=action_utilities.get)
            # print("taking " + str(best_action))
        else:
            best_action = Directions.STOP  # Default action if no legal moves

        return api.makeMove(best_action, legal)
