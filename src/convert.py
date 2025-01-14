#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:09:38 2025

@author: kevin
"""

import pickle
from train import NeuralNetwork

path = "/home/kevin/Documents/ProjetRL2025/mva-rl-assignment-KevinHOARAU974-backup/src/DQN_8.pkl"

with open(path, 'rb') as f:
            loaded_data = pickle.load(f)
            
param_nn = loaded_data["param_nn"]
nb_actions = loaded_data["nb_actions"]
state_dimension = loaded_data["state_dimension"]
nb_neurons = loaded_data["nb_neurons"]

DQN = NeuralNetwork(state_dimension, nb_neurons, nb_actions)
DQN.load_state_dict(param_nn)
DQN = DQN.to("cpu")

data_to_save = {"nb_actions": nb_actions,
                "state_dimension": state_dimension,
                "nb_neurons": nb_neurons,
                "param_nn": DQN.state_dict()}

path = "/home/kevin/Documents/ProjetRL2025/mva-rl-assignment-KevinHOARAU974/src/DQN_8.pkl"
with open(path, 'wb') as f:
    pickle.dump(data_to_save, f)

