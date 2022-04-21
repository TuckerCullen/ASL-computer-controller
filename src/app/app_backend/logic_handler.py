
import control_functions
from control_functions import *

ACTION_LOOKUP = {
	22: "Check Weather", 
	25 : "Take Screenshot",
	2 : "Take Picture", 
	31 : "Open Browser"
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
		
		if cur_action == "Check Weather":
			control_functions.check_weather()

		if cur_action == "Open Browser":
			control_functions.open_browser()

		action_log.append(cur_action)
	
	def get_commands(self):
		# return slider_state+input_state
		return self.input_state

