from calendar import c


from glob import glob
from pyexpat import model
import torch
from pyexpat import model
import torch
import control_functions
from computer_controls import control_functions
from control_functions import *

import pandas as pd
import ast
import numpy as np
import json

import sys
sys.path.append('../')

from app import get_feature
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
			model.load_state_dict(torch.load("model_checkpoints/" + model_name + ".pth"))
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


# Dictionary Mapping
# action_dic = {"computer": 0, "want": 1, "play": 2, "clothes": 3, "tell": 4, "now": 5, "deaf": 6, "many": 7, "before": 8, "pink": 9, "give": 10, "short": 11, "graduate": 12, "cool": 13, "son": 14, "kiss": 15, "forget": 16, "hat": 17, "school": 18, "purple": 19, "drink": 20, "apple": 21, "table": 22, "orange": 23, "thursday": 24, "basketball": 25, "secretary": 26, "go": 27, "corn": 28, "fish": 29, "finish": 30, "book": 31, "yes": 32, "white": 33, "eat": 34, "cook": 35, "paint": 36, "tall": 37, "enjoy": 38, "meet": 39, "brown": 40, "time": 41, "woman": 42, "cousin": 43, "fine": 44, "same": 45, "yellow": 46, "wife": 47, "family": 48, "all": 49, "help": 50, "wrong": 51, "walk": 52, "pizza": 53, "decide": 54, "wait": 55, "no": 56, "hearing": 57, "but": 58, "dark": 59, "can": 60, "man": 61, "bird": 62, "bed": 63, "doctor": 64, "black": 65, "right": 66, "shirt": 67, "like": 68, "cow": 69, "medicine": 70, "jacket": 71, "study": 72, "cheat": 73, "blue": 74, "mother": 75, "candy": 76, "language": 77, "year": 78, "thin": 79, "what": 80, "birthday": 81, "chair": 82, "accident": 83, "letter": 84, "thanksgiving": 85, "who": 86, "need": 87, "how": 88, "africa": 89, "dog": 90, "later": 91, "bowling": 92, "color": 93, "paper": 94, "change": 95, "hot": 96, "last": 97, "dance": 98, "work": 99}
action_dic = { 63 : "bed"}
slider_dic = { 27 : "go"}

def model_to_command(model_output):
	# if command is in the action dic, then store into the input_state
	# if command is in the slider state, then store into the slider_state
	global input_state
	global slider_state
	key = model_output
	if action_dic.has_key(key):
    	input_state = action_dic[key]
	if slider_dic.has_key(key):
    	slider_state = slider_dic[key]

# return commands to the front-end
def get_commands(slider_state,input_state):
    return slider_state, input_state

#processes running list of actions into corresponding functions. TODO: Use Stream optimal logic
def action_stream_handler(action_log):

# Dictionary Mapping
# action_dic = {"computer": 0, "want": 1, "play": 2, "clothes": 3, "tell": 4, "now": 5, "deaf": 6, "many": 7, "before": 8, "pink": 9, "give": 10, "short": 11, "graduate": 12, "cool": 13, "son": 14, "kiss": 15, "forget": 16, "hat": 17, "school": 18, "purple": 19, "drink": 20, "apple": 21, "table": 22, "orange": 23, "thursday": 24, "basketball": 25, "secretary": 26, "go": 27, "corn": 28, "fish": 29, "finish": 30, "book": 31, "yes": 32, "white": 33, "eat": 34, "cook": 35, "paint": 36, "tall": 37, "enjoy": 38, "meet": 39, "brown": 40, "time": 41, "woman": 42, "cousin": 43, "fine": 44, "same": 45, "yellow": 46, "wife": 47, "family": 48, "all": 49, "help": 50, "wrong": 51, "walk": 52, "pizza": 53, "decide": 54, "wait": 55, "no": 56, "hearing": 57, "but": 58, "dark": 59, "can": 60, "man": 61, "bird": 62, "bed": 63, "doctor": 64, "black": 65, "right": 66, "shirt": 67, "like": 68, "cow": 69, "medicine": 70, "jacket": 71, "study": 72, "cheat": 73, "blue": 74, "mother": 75, "candy": 76, "language": 77, "year": 78, "thin": 79, "what": 80, "birthday": 81, "chair": 82, "accident": 83, "letter": 84, "thanksgiving": 85, "who": 86, "need": 87, "how": 88, "africa": 89, "dog": 90, "later": 91, "bowling": 92, "color": 93, "paper": 94, "change": 95, "hot": 96, "last": 97, "dance": 98, "work": 99}
action_dic = { 63 : "bed"}
slider_dic = { 27 : "go"}

def model_to_command(model_output):
	# if command is in the action dic, then store into the input_state
	# if command is in the slider state, then store into the slider_state
	global input_state
	global slider_state
	key = model_output
	if action_dic.has_key(key):
		input_state = action_dic[key]
	if slider_dic.has_key(key):
		slider_state = slider_dic[key]

# return commands to the front-end
def get_commands():
	global input_state, slider_state
	return "open chrome"
	# return slider_state+input_state

#processes running list of actions into corresponding functions. TODO: Use Stream optimal logic
def action_stream_handler(action_log):

	global input_state,slider_state

	#use state variables to keep track of relevant checks

	#input_states -> action that will require a future input
	#slider_state -> action that rquires a future sliding direction

	#cur_action = action_log[-1]
	cur_action = input_state

	if slider_state == "take" and cur_action == "picture":
		control_functions.take_picture()

	if slider_state == "take" and cur_action == "screenshot":
		control_functions.screenshot()
	
	if slider_state == "check" and cur_action == "weather":
		control_functions.check_weather()
	
	print("STATES", input_state, slider_state)

	# if slider_state == "brightness" and cur_action == "up":
    # 	control_functions.brightness()

	# if slider_state == "volume" and cur_action == "up":
	# 	control_functions.volume(True) 

	# if slider_state == "volume" and cur_action == "down":
	# 	control_functions.volume(False)

	# if cur_action != "none" and input_state == "open":
	# 	print("OPEN: ", cur_action)
	# 	control_functions.open(cur_action)

	# if (cur_action == "volume"):
	# 	slider_state = "volume"

	# if (cur_action == "open"):
	# 	input_state = "open"
	
	# print("STATES", input_state, slider_state)



#runs the assistant and passes the execution to relevant functions
def run_assistant(live = True):
	model = get_model("model1")
	active = True

	#actions
	action_log = []

	if (live):
		while(active):
			# prompt = input("Type h for help, q to quit").lower()

			#prompt handler
			# if (prompt == "q"):
			# 	active = False
			# 	break
			# action_appended = False

			print(get_feature())

			# data = get_data_live(120)

			# action = get_result(model, 'mini_train.csv')


			# if len(action_log) == 0 or action != action_log[-1]:
			# 	action_log.append(action)
			# 	action_appended = True

			# if action_appended:
			# 	action_stream_handler(action_log)

			# print(action_log, action_appended)


			



# m = get_model("model1")

# get_result(m, 'mini_train.csv')

# run_assistant()