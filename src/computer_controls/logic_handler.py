from control_functions import *
from spoter.utils import evaluate

# Returns the model to use for evaluation
def get_model(model_name, use_cached = True):

	if (use_cached):
		#load model_name and return it
		return None

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

action_lookup = {}

#return string version of output from model
def get_result():

	inputs, labels = data
	inputs = inputs.squeeze(0).to(device)
	labels = labels.to(device, dtype=torch.long)
	outputs = model(inputs).expand(1, -1, -1)

	action_int = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))	

	return action_lookup[action_int]


#processes running list of actions into corresponding functions. TODO: Use Stream optimal logic
def action_stream_handler(action_log):
	#use state variables to keep track of relevant checks

	#input_states -> action that will require a future input
	#slider_state -> action that rquires a future sliding direction

	cur_action = action_log[-1]

	if cur_action == "up" and slider_state == "volume":
		control_functions.volume(True) 

	if cur_action == "down" and slider_state == "volume":
		control_functions.volume(False)

	if cur_action != "none" and input_state == "open":
		control_functions.open(cur_action)

	if (cur_action == "volume"):
		slider_state = "volume"

	if (cur_action == "open"):
		input_state == "open"



#runs the assistant and passes the execution to relevant functions
def run_assistant(live = True):
	model = get_model()
	active = True

	#actions
	action_log = []

	if (live):
		while(active):
			prompt = input("Type h for help, q to quit").lower()

			#prompt handler
			if (prompt == "q"):
				active = False
				break

			data = get_data_live(120)

			action = get_result()

			if action != action_log[-1]:
				action_log.append(action)
				action_appended = True

			if action_appended:
				action_stream_handler(action_log)








