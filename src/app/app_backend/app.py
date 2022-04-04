from concurrent.futures import process
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

#Set up Flask:
app = Flask(__name__)
#Set up Flask to bypass CORS:
cors = CORS(app)

features = []
per_frame_feature = []
#Create the receiver API POST endpoint:
@app.route("/receiver", methods=["POST"])
def postME():
    global features, per_frame_feature

    print("HIT")

    data = request.get_json()
    per_frame_feature, features = process_data(data, features)
    return {
        'statusCode': 200,
    }

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
    print("feature", features)
    return cleaner(features)

def get_per_frame_feature():
    return per_frame_feature

if __name__ == "__main__": 
    app.run(debug=True)