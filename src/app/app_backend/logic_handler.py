
import control_functions
from control_functions import *

ACTION_LOOKUP = {
	# Most consistantly classified funcs:
	22: "Check Weather (gloss: Weather)" , 
	85 : "Open Browser (gloss: Open)",
	5 : "Open Twitter (gloss: Bird)",
	66 : "Sleep (gloss: Dark)",
	
	# less consistantly classfified 
	13 : "Volume Up (gloss: Loud)",
	68 : "Volume Down (gloss: Down)",

	# not totally working
	4 : "Quit (gloss: Cancel)", 

	# no mapping function yet: 
	101 : "Take Screenshot", 
	101 : "Take Picture",
}

class LogicHandler:

	def __init__(self):
		
		self.input_state = None
		self.action_log = []


	def model_to_command(self, model_result: list):
		"""
		Called in app.py to pass model output to logic handler. 
		
		Checks to see if the result of the model is an action, and calls action_stream_handler to process action
		"""
		#gloss = model_result
		self.input_state = None
		for label in model_result:
			if label in ACTION_LOOKUP:
				self.input_state = ACTION_LOOKUP[label] 
				print("INPUT STATE: ", self.input_state)
				self.action_stream_handler(self.action_log)
				break


	#processes running list of actions into corresponding functions. TODO: Use Stream optimal logic
	def action_stream_handler(self, action_log):
		"""
		Calls the proper control function based on the input_state. 
		"""

		cur_action = self.input_state

		if cur_action == "Take Picture":
			control_functions.take_picture()

		if cur_action == "Take Screenshot":
			control_functions.screenshot()
		
		if cur_action == "Check Weather (gloss: Weather)":
			control_functions.check_weather()

		if cur_action == "Open Browser (gloss: Open)":
			control_functions.open_browser()

		if cur_action == "Open Twitter (gloss: Bird)":
			control_functions.open_twitter()
		
		if cur_action == "Sleep (gloss: Dark)":
			control_functions.sleep()

		if cur_action == "Volume Up / Loud":
			control_functions.volume(direction="UP")

		if cur_action == "Volume Down (gloss: Down)":
			control_functions.volume(direction="DOWN")

		if cur_action == "Quit (gloss: Cancel)":
			control_functions.cancel()

		action_log.append(cur_action)
	
	def get_commands(self):
		# return slider_state+input_state
		return self.input_state

