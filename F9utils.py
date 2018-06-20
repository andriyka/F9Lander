# -------------------------------------------------- #
# --------------------_F9_Lander_------------------- #
# ----------------------WRAPPER--------------------- #
# -------------------------------------------------- #
# imports
from __future__ import print_function
import socket
import pickle
import json
import glob
import os

# -------------------------------------------------- #

RESET_CMD = [0, 0, 0, 1]
DEFAULT_IP = '127.0.0.1'
DEFAULT_PORT = 50007

# -------------------------------------------------- #


class F9GameClient:
    """Game wrapper"""
    def __init__(self, ip=DEFAULT_IP, port=DEFAULT_PORT):
        self.socket = socket.socket()
        self.socket.connect((ip, port))
        self.curState = None
        self.reset_game()
        self.totalScore = 0
        self.vy_hist = []

    def send(self, message):
        self.socket.send(json.dumps(message).encode('utf-8'))

    def reset_game(self):
        # send init command | new game
        self.send(RESET_CMD)
        self.curState = self.getServerState()

    def isTerminalState(self, state):
        # system["flight_status"] | "none", "landed", "destroyed"
        # "none" means that we don't know, whether we landed or destroyed
        agent, _, system = state
        status = False
        if system["flight_status"] == "destroyed" or system["flight_status"] == "landed" or agent["py"] <= 0.0:
            status = True
        return status

    def getReward(self, state):
        agent, platform, system = state
        if system["flight_status"] == "landed":
            score = 100.0
        elif self.isTerminalState(state):
            # MAYBE BUG ---> delete "or system["flight_status"] == "landed" from isTerminalState
            # but elif fixed it, so it's feature
            score = -100.0
        else:
            # --------- YOUR CODE HERE ----------
            # You can write a reward function here. It will be used as a heuristics
            # for the states when the rocket is neither landed nor crashed.
            # if self.agent_dist_history:
            #     score += (self.agent_dist_history[-1] - agent['dist'])



            # The following logic provides scoring based on main conditions for successful landing: speed,
            # angle and position in relation to the platform
            # if two of three indicators are positive - the score is positive but it's dynamics(up, down) depends on the
            # third indicator, so the brain could be able to determine how to tune the vy speed
            score = 0.0

            score += 30 if (7.0 + agent["vy"]) > 0 else 0# Do nothing

            if self.vy_hist:
                score += (abs(self.vy_hist[-1]) - abs(agent['vy'])) * 10 # vy should decrease

            self.vy_hist.append(agent['vy'])

            if agent["angle"] < 0.01:
                score += 5
            else:
                score -= agent["angle"] * 10

            if abs(agent['px'] - platform['px']) < 15:
                score += 5
            else:
                score -= abs(agent['px'] - platform['px']) / 3

            if agent['contact'] and abs(agent['vy']) < 7:
                score += 50

            # -----------------------------------

        self.totalScore += score
        return score

    def actions(self, state=None):
        # returns legal actions, keys map [up, left, right, reset_game]
        act = [[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0], [1,1,0,0], [0,1,1,0], [1,0,1,0], [1,1,1,0]]
        return act

    def getServerState(self):
        # getting data from server
        data = eval(self.socket.recv(1024))
        state = None
        if data:
            agent_state = next(item for item in data if item["type"] == "actor")
            platform_state = next(item for item in data if item["type"] == "decoration")
            system_state = next(item for item in data if item["type"] == "system")
            state = [agent_state, platform_state, system_state]
        return state

    def doAction(self, action):
        # act in the game environment
        if any([action == act for act in self.actions()]):
            self.send(action)
            self.curState = self.getServerState()
        else:
            print("Invalid Action")

# -------------------------------------------------- #


class RLAgent:
    """Abstract class: an RLAgent performs reinforcement learning.
    The game client will call getAction() to get an action, perform the action, and
    then provide feedback (via provideFeedback()) to the RL algorithm, so it can learn."""
    def getAction(self, state):
        raise NotImplementedError("Override me")

    def provideFeedback(self, state, action, reward, new_state):
        raise NotImplementedError("Override me")

# -------------------------------------------------- #


class Snapshot:
    def __init__(self, prefix):
        self.prefix = prefix

    def save(self, state, num):
        file_path = '_'.join([self.prefix, str(num)])
        file_name = ''.join([file_path, '.pkl'])
        pickle.dump(state, file_name, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        file_path = glob.glob(''.join([self.prefix, "*_[0-9]*.pkl"]))
        file_path.sort(key=os.path.getctime)
        if not len(file_path):
            return

        print("Loading snapshot", file_path[-1])
        return pickle.load(file_path[-1])
