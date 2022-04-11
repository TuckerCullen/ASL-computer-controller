from calendar import c
import torch
# from control_functions import *

import pandas as pd
import ast
import numpy as np
import json

# import sys
# sys.path.append('../')
# sys.path.append('../../')

from ML_models.spoter.spoter.spoter_model import SPOTER
# from ML_models.spoter.datasets.czech_slr_dataset import CzechSLRDataset
from torch.utils.data import DataLoader

BODY_IDENTIFIERS = [
    "nose",
    "neck",
    "rightEye",
    "leftEye",
    "rightEar",
    "leftEar",
    "rightShoulder",
    "leftShoulder",
    "rightElbow",
    "leftElbow",
    "rightWrist",
    "leftWrist"
]

HAND_IDENTIFIERS = [
    "wrist",
    "indexTip",
    "indexDIP",
    "indexPIP",
    "indexMCP",
    "middleTip",
    "middleDIP",
    "middlePIP",
    "middleMCP",
    "ringTip",
    "ringDIP",
    "ringPIP",
    "ringMCP",
    "littleTip",
    "littleDIP",
    "littlePIP",
    "littleMCP",
    "thumbTip",
    "thumbIP",
    "thumbMP",
    "thumbCMC"
]

HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]



# from spoter.utils import evaluate

# Returns the model to use for evaluation

num_classes = 100
hidden_dim = 108





def get_model(model_name, use_cached = True):

	if (use_cached):
		#load model_name and return it

			model = SPOTER(num_classes=num_classes, hidden_dim=hidden_dim)
			# tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
			model.load_state_dict(torch.load(model_name + ".pth"))
			model.train(False)
			return model                
	else:
		#train a new model and return it
		return None

#returns the data to use for evaluation from file
def get_data_file(file_name):

	#load the images/video from the file and return them
	return None

#returns the data to use for evaluation from live webcam
def get_data_live(num_frames):
	#open webcam and capture num_frames and return them
  
	data_preprocessing()

	return None

action_lookup = {28: "open"}

#return string version of output from model
def get_result(model, inputs):

	# mini = pd.read_csv('mini_train.csv')

	g = torch.Generator()
	device = torch.device("cpu")

	mini_data = CzechSLRDataset(inputs)
	mini_loader = DataLoader(mini_data, shuffle=False, generator=g)	

	for i, data in enumerate(mini_loader):
		inputs, labels = data
		inputs = inputs.squeeze(0).to(device)
		labels = labels.to(device, dtype=torch.long)
		
		outputs = model(inputs).expand(1, -1, -1)
		
		result = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))
		correct = int(labels[0][0])

		print("EVALUATE", result, correct)

		#BREAK SO IT ONLY DOES ONE
		break

	return action_lookup[result]

	
	return


input_state = None
slider_state = None

#processes running list of actions into corresponding functions. TODO: Use Stream optimal logic
def action_stream_handler(action_log):

	global input_state
	global slider_state

	#use state variables to keep track of relevant checks

	#input_states -> action that will require a future input
	#slider_state -> action that rquires a future sliding direction

	cur_action = action_log[-1]

	if cur_action == "up" and slider_state == "volume":
		control_functions.volume(True) 

	if cur_action == "down" and slider_state == "volume":
		control_functions.volume(False)

	if cur_action != "none" and input_state == "open":
		print("OPEN: ", cur_action)
		control_functions.open(cur_action)

	if (cur_action == "volume"):
		slider_state = "volume"

	if (cur_action == "open"):
		input_state = "open"
	
	print("STATES", input_state, slider_state)



#runs the assistant and passes the execution to relevant functions
def run_assistant(live = True):
	model = get_model("model1")
	active = True

	#actions
	action_log = []

	if (live):
		while(active):
			prompt = input("Type h for help, q to quit").lower()

			# prompt handler
			if (prompt == "q"):
				active = False
				break
			action_appended = False

			# print(get_feature())

			# data = get_data_live(120)

			action = get_result(model, 'mini_train.csv')


			if len(action_log) == 0 or action != action_log[-1]:
				action_log.append(action)
				action_appended = True

			if action_appended:
				action_stream_handler(action_log)

			print(action_log, action_appended)


			



# m = get_model("model1")

# get_result(m, 'mini_train.csv')

# run_assistant()