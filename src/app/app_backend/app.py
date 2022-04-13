from pickle import FALSE
import sys
sys.path.append('../../')
sys.path.append('../')

from concurrent.futures import process
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import logic_handler as lh
import logging

from ML_models import *
from ML_models.spoter.spoter.spoter_model import SPOTER
from ML_models.spoter.normalization import *
from ML_models.spoter.normalization.body_normalization import BODY_IDENTIFIERS
from ML_models.spoter.normalization.hand_normalization import HAND_IDENTIFIERS
import torch
import torch.utils.data as torch_data
from torch.utils.data import DataLoader
import ast

HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]

def load_dataset(file_location: str):

    # print("FILE", file_location)

    # Load the dataset csv file

    #CHANGED TO DIRECTLY FILE_LOCATION SINCE PASSED AS PANDAS
    # print(type(file_location))

    # df = pd.read_csv(file_location, encoding="utf-8")

    df = file_location

    # TO BE DELETED
    df.columns = [item.replace("_left_", "_0_").replace("_right_", "_1_") for item in list(df.columns)]
    if "neck_X" not in df.columns:
        df["neck_X"] = [0 for _ in range(df.shape[0])]
        df["neck_Y"] = [0 for _ in range(df.shape[0])]

    print("FRAMES USED: ", df.shape[0])

    #CHANGED TO MAKE EMPTY LABELS
    labels = [-1] * df.shape[0]
    data = []

    for row_index, row in df.iterrows():
        # original type is str, to get the length, use literal_eval to convert str to a list
        current_row = np.empty(shape=(len(ast.literal_eval(row["leftEar_X"])), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))
        for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            current_row[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            current_row[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])

        data.append(current_row)  # current_row shape: # of frames(frames in a video) * keypoints_index (54 keypoints) * 2 (x and y)
    return data, labels


def tensor_to_dictionary(landmarks_tensor: torch.Tensor) -> dict:

    data_array = landmarks_tensor.numpy()
    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index]

    return output


def dictionary_to_tensor(landmarks_dict: dict) -> torch.Tensor:

    output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return torch.from_numpy(output)


class CzechSLRDataset(torch_data.Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, dataset_filename: str, num_labels=5, transform=None, augmentations=False,
                 augmentations_prob=0.5, normalize=False):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """

        loaded_data = load_dataset(dataset_filename)
        data, labels = loaded_data[0], loaded_data[1]

        self.data = data
        self.labels = labels
        self.targets = list(labels)
        self.num_labels = num_labels
        self.transform = transform

        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """

        depth_map = torch.from_numpy(np.copy(self.data[idx]))
        label = torch.Tensor([self.labels[idx] - 1])

        depth_map = tensor_to_dictionary(depth_map)

        # Apply potential augmentations
        if self.augmentations and random.random() < self.augmentations_prob:

            selected_aug = randrange(4)

            if selected_aug == 0:
                depth_map = augment_rotate(depth_map, (-13, 13))

            if selected_aug == 1:
                depth_map = augment_shear(depth_map, "perspective", (0, 0.1))

            if selected_aug == 2:
                depth_map = augment_shear(depth_map, "squeeze", (0, 0.15))

            if selected_aug == 3:
                depth_map = augment_arm_joint_rotate(depth_map, 0.3, (-4, 4))

        if self.normalize:
            depth_map = normalize_single_body_dict(depth_map)
            depth_map = normalize_single_hand_dict(depth_map)

        depth_map = dictionary_to_tensor(depth_map)

        # Move the landmark position interval to improve performance
        depth_map = depth_map - 0.5

        if self.transform:
            depth_map = self.transform(depth_map)

        return depth_map, label

    def __len__(self):
        return len(self.labels)

#Set up Flask:
app = Flask(__name__)
#Set up Flask to bypass CORS:
cors = CORS(app)

features = []
per_frame_feature = []


def get_model(model_name, use_cached = True):

    num_classes = 100
    hidden_dim = 108

    if (use_cached):

        #load model_name and return it

            model = SPOTER(num_classes=num_classes, hidden_dim=hidden_dim)
            # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
            model.load_state_dict(torch.load("./model_checkpoints/" + model_name + ".pth"))
            model.train(False)
            return model                
    else:
        #train a new model and return it
        return None



m = get_model("model1")
prev_r = None

#Create the receiver API POST endpoint:
@app.route("/receiver", methods=["POST"])
def postME():
    global features, per_frame_feature, prev_r

    # print("H")

    data = request.get_json()
    per_frame_feature, features = process_data(data, features)

    # print(m)
    print(features)

    r = get_result(m, get_feature())

    print("RESULT: ", r)

    prev_r = r

    return {
        'statusCode': 200,
    }

@app.route("/sender",methods=["GET"])
def getME():
    print("test111")
    message = {'Prediction': lh.get_commands()}
    return jsonify(message)  
    #return lh.get_commands()


def process_data(results, features):
    data = [[0]]        ### add a id slot, only for later groupby function in pandas
    # left hand 21

    if 'leftHandLandmarks' in results:
        for data_points in results['leftHandLandmarks']:
            data.append([data_points['x']])
            data.append([data_points['y']])
    else:
        for i in range(21):
            data.append([0])
            data.append([0])

    # right hand 21
    if 'rightHandLandmarks' in results:                     
        for data_points in results['rightHandLandmarks']:
            data.append([data_points['x']])
            data.append([data_points['y']])
    else:
        for i in range(21):
            data.append([0])
            data.append([0])


    # body 33
    if 'poseLandmarks' in results != None:
        for data_points in results['poseLandmarks']:
            data.append([data_points['x']])
            data.append([data_points['y']])
    else:
        for i in range(33):
            data.append([0])
            data.append([0])
    
    if len(features) == 0:
        features = np.array(data)
    else:
        features = np.hstack((features, data))              # shape: 150 * number of frames

    return data, features


def cleaner(data):
    header = ['id',
          'wrist_left_X', 'wrist_left_Y',
          'thumbCMC_left_X', 'thumbCMC_left_Y', 'thumbMP_left_X', 'thumbMP_left_Y', 'thumbIP_left_X', 'thumbIP_left_Y', 'thumbTip_left_X', 'thumbTip_left_Y', 
          'indexMCP_left_X', 'indexMCP_left_Y', 'indexPIP_left_X', 'indexPIP_left_Y', 'indexDIP_left_X', 'indexDIP_left_Y', 'indexTip_left_X', 'indexTip_left_Y',
          'middleMCP_left_X', 'middleMCP_left_Y', 'middlePIP_left_X', 'middlePIP_left_Y', 'middleDIP_left_X', 'middleDIP_left_Y', 'middleTip_left_X', 'middleTip_left_Y', 
          'ringMCP_left_X', 'ringMCP_left_Y', 'ringPIP_left_X', 'ringPIP_left_Y', 'ringDIP_left_X','ringDIP_left_Y', 'ringTip_left_X', 'ringTip_left_Y',
          'littleMCP_left_X', 'littleMCP_left_Y', 'littlePIP_left_X', 'littlePIP_left_Y', 'littleDIP_left_X', 'littleDIP_left_Y', 'littleTip_left_X', 'littleTip_left_Y',
          'wrist_right_X', 'wrist_right_Y', 
          'thumbCMC_right_X', 'thumbCMC_right_Y', 'thumbMP_right_X',  'thumbMP_right_Y', 'thumbIP_right_X', 'thumbIP_right_Y', 'thumbTip_right_X', 'thumbTip_right_Y',
          'indexMCP_right_X', 'indexMCP_right_Y', 'indexPIP_right_X', 'indexPIP_right_Y', 'indexDIP_right_X', 'indexDIP_right_Y', 'indexTip_right_X', 'indexTip_right_Y',
          'middleMCP_right_X', 'middleMCP_right_Y', 'middlePIP_right_X',  'middlePIP_right_Y', 'middleDIP_right_X', 'middleDIP_right_Y', 'middleTip_right_X',  'middleTip_right_Y',
          'ringMCP_right_X', 'ringMCP_right_Y', 'ringPIP_right_X', 'ringPIP_right_Y', 'ringDIP_right_X', 'ringDIP_right_Y', 'ringTip_right_X', 'ringTip_right_Y',
          'littleMCP_right_X', 'littleMCP_right_Y', 'littlePIP_right_X',  'littlePIP_right_Y', 'littleDIP_right_X', 'littleDIP_right_Y', 'littleTip_right_X', 'littleTip_right_Y',
          'nose_X', 'nose_Y',
          'leftEye_inner_X', 'leftEye_inner_Y', 'leftEye_X', 'leftEye_Y', 'leftEye_outer_X', 'leftEye_outer_Y', 
          'rightEye_inner_X', 'rightEye_inner_Y', 'rightEye_X', 'rightEye_Y', 'rightEye_outer_X', 'rightEye_outer_Y',
          'leftEar_X', 'leftEar_Y', 'rightEar_X', 'rightEar_Y',
          'mouthLeft_X', 'mouthLeft_Y', 'mouthRight_X', 'mouthRight_Y',
          'leftShoulder_X', 'leftShoulder_Y', 'rightShoulder_X', 'rightShoulder_Y',
          'leftElbow_X', 'leftElbow_Y', 'rightElbow_X', 'rightElbow_Y',
          'leftWrist_X', 'leftWrist_Y', 'rightWrist_X', 'rightWrist_Y',
          'leftPinky_X', 'leftPinky_Y', 'rightPinky_X', 'rightPinky_Y', 'leftIndex_X', 'leftIndex_Y', 'rightIndex_X', 'rightIndex_Y', 
          'leftThumb_X', 'leftThumb_Y', 'rightThumb_X', 'rightThumb_Y', 'leftHip_X', 'leftHip_Y', 'rightHip_X', 'rightHip_Y', 
          'leftKnee_X', 'leftKnee_Y', 'rightKnee_X', 'rightKnee_Y', 'leftAnkle_X', 'leftAnkle_Y', 'rightAnkle_X', 'rightAnkle_Y', 
          'leftHeel_X', 'leftHeel_Y', 'rightHeel_X', 'rightHeel_Y', 'leftFootIndex_X', 'leftFootIndex_Y', 'rightFootIndex_X', 'rightFootIndex_Y']
    gold_columns = ['id', 'wrist_left_X',
                    'wrist_left_Y',
                    'thumbCMC_left_X',
                    'thumbCMC_left_Y',
                    'thumbMP_left_X',
                    'thumbMP_left_Y',
                    'thumbIP_left_X',
                    'thumbIP_left_Y',
                    'thumbTip_left_X',
                    'thumbTip_left_Y',
                    'indexMCP_left_X',
                    'indexMCP_left_Y',
                    'indexPIP_left_X',
                    'indexPIP_left_Y',
                    'indexDIP_left_X',
                    'indexDIP_left_Y',
                    'indexTip_left_X',
                    'indexTip_left_Y',
                    'middleMCP_left_X',
                    'middleMCP_left_Y',
                    'middlePIP_left_X',
                    'middlePIP_left_Y',
                    'middleDIP_left_X',
                    'middleDIP_left_Y',
                    'middleTip_left_X',
                    'middleTip_left_Y',
                    'ringMCP_left_X',
                    'ringMCP_left_Y',
                    'ringPIP_left_X',
                    'ringPIP_left_Y',
                    'ringDIP_left_X',
                    'ringDIP_left_Y',
                    'ringTip_left_X',
                    'ringTip_left_Y',
                    'littleMCP_left_X',
                    'littleMCP_left_Y',
                    'littlePIP_left_X',
                    'littlePIP_left_Y',
                    'littleDIP_left_X',
                    'littleDIP_left_Y',
                    'littleTip_left_X',
                    'littleTip_left_Y',
                    'wrist_right_X',
                    'wrist_right_Y',
                    'thumbCMC_right_X',
                    'thumbCMC_right_Y',
                    'thumbMP_right_X',
                    'thumbMP_right_Y',
                    'thumbIP_right_X',
                    'thumbIP_right_Y',
                    'thumbTip_right_X',
                    'thumbTip_right_Y',
                    'indexMCP_right_X',
                    'indexMCP_right_Y',
                    'indexPIP_right_X',
                    'indexPIP_right_Y',
                    'indexDIP_right_X',
                    'indexDIP_right_Y',
                    'indexTip_right_X',
                    'indexTip_right_Y',
                    'middleMCP_right_X',
                    'middleMCP_right_Y',
                    'middlePIP_right_X',
                    'middlePIP_right_Y',
                    'middleDIP_right_X',
                    'middleDIP_right_Y',
                    'middleTip_right_X',
                    'middleTip_right_Y',
                    'ringMCP_right_X',
                    'ringMCP_right_Y',
                    'ringPIP_right_X',
                    'ringPIP_right_Y',
                    'ringDIP_right_X',
                    'ringDIP_right_Y',
                    'ringTip_right_X',
                    'ringTip_right_Y',
                    'littleMCP_right_X',
                    'littleMCP_right_Y',
                    'littlePIP_right_X',
                    'littlePIP_right_Y',
                    'littleDIP_right_X',
                    'littleDIP_right_Y',
                    'littleTip_right_X',
                    'littleTip_right_Y',
                    'nose_X',
                    'nose_Y',
                    'leftEye_X',
                    'leftEye_Y',
                    'rightEye_X',
                    'rightEye_Y',
                    'leftEar_X',
                    'leftEar_Y',
                    'rightEar_X',
                    'rightEar_Y',
                    'leftShoulder_X',
                    'leftShoulder_Y',
                    'rightShoulder_X',
                    'rightShoulder_Y',
                    'leftElbow_X',
                    'leftElbow_Y',
                    'rightElbow_X',
                    'rightElbow_Y',
                    'leftWrist_X',
                    'leftWrist_Y',
                    'rightWrist_X',
                    'rightWrist_Y',
                    'neck_X',
                    'neck_Y']
    
    df = pd.DataFrame(data.transpose(), columns = header)
    # Remove unused columns
    unused_idx = [col not in gold_columns for col in list(df.columns)]
    unused_columns = df.columns[unused_idx]
    df = df.drop(unused_columns, axis=1)
    
    # Add Missed column
    df['neck_X'] = (df['leftShoulder_X'] + df['rightShoulder_X']) / 2
    df['neck_Y'] = (df['leftShoulder_Y'] + df['rightShoulder_Y']) / 2

    df_list = df.groupby(['id']).agg(lambda x: str(list(x)))

    return df_list

def get_feature():
    # print("feature", features)
    return cleaner(features)

def get_per_frame_feature():
    return per_frame_feature


action_lookup = {0: 'computer', 1: 'want', 2: 'play', 3: 'clothes', 4: 'tell', 5: 'now', 6: 'deaf', 7: 'many', 8: 'before', 9: 'pink', 10: 'give', 11: 'short', 12: 'graduate', 13: 'cool', 14: 'son', 15: 'kiss', 16: 'forget', 17: 'hat', 18: 'school', 19: 'purple', 20: 'drink', 21: 'apple', 22: 'table', 23: 'orange', 24: 'thursday', 25: 'basketball', 26: 'secretary', 27: 'go', 28: 'corn', 29: 'fish', 30: 'finish', 31: 'book', 32: 'yes', 33: 'white', 34: 'eat', 35: 'cook', 36: 'paint', 37: 'tall', 38: 'enjoy', 39: 'meet', 40: 'brown', 41: 'time', 42: 'woman', 43: 'cousin', 44: 'fine', 45: 'same', 46: 'yellow', 47: 'wife', 48: 'family', 49: 'all', 50: 'help', 51: 'wrong', 52: 'walk', 53: 'pizza', 54: 'decide', 55: 'wait', 56: 'no', 57: 'hearing', 58: 'but', 59: 'dark', 60: 'can', 61: 'man', 62: 'bird', 63: 'bed', 64: 'doctor', 65: 'black', 66: 'right', 67: 'shirt', 68: 'like', 69: 'cow', 70: 'medicine', 71: 'jacket', 72: 'study', 73: 'cheat', 74: 'blue', 75: 'mother', 76: 'candy', 77: 'language', 78: 'year', 79: 'thin', 80: 'what', 81: 'birthday', 82: 'chair', 83: 'accident', 84: 'letter', 85: 'thanksgiving', 86: 'who', 87: 'need', 88: 'how', 89: 'africa', 90: 'dog', 91: 'later', 92: 'bowling', 93: 'color', 94: 'paper', 95: 'change', 96: 'hot', 97: 'last', 98: 'dance', 99: 'work'}

def get_result(model, inputs):

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

        # print("NUMERIC EVAL: ", result)

        #BREAK SO IT ONLY DOES ONE
        break
    
    if result in action_lookup:
        return action_lookup[result]
    else:
        return "NO_ACTION"


if __name__ == "__main__": 
    logging.getLogger('werkzeug').disabled = True
    app.run(debug=True)