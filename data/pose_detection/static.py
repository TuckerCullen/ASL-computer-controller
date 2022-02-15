import cv2
import mediapipe as mp
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

#  create a CSV file
header = ['gloss', 'video_id', 'frame_id',
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
          'Nose_X', 'Nose_Y',
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

with open(r'C:\Projects\ATCC-SLR\Data-SLR\raw.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # read all images in the data folder
    root_dir = r'C:\Projects\ATCC-SLR\Data-SLR\data'            # chunked img from videos
    des_dir = r'C:\Projects\ATCC-SLR\Data-SLR\keypoints'
    try:
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)
    except OSError:
        print('Error: Creating directory of data')

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            data = []
            image_path = os.path.join(subdir, file)
            
            classname = subdir.split("\\")[-2]
            video_id = subdir.split("\\")[-1]
            data.append(classname)
            data.append(video_id)

            name = file.split('.jpg')[0]
            data.append(name)

            store_path = os.path.join(des_dir, classname)
            try:
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
            except OSError:
                print('Error: Creating directory of data')

            store_path = os.path.join(store_path, video_id)
            try:
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
            except OSError:
                print('Error: Creating directory of data')

            store_path = os.path.join(store_path, name + '_kp' + '.jpg')
            print("saving keypoints for " + store_path)

            with mp_holistic.Holistic(static_image_mode=True) as holistic:
                image = cv2.imread(image_path)
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                imageHeight, imageWidth, _ = image.shape
                
                # drawing for testing
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # left hand 21
                if results.left_hand_landmarks != None:
                    for data_points in results.left_hand_landmarks.landmark:
                        data.append(data_points.x)
                        data.append(data_points.y)
                else:
                    for i in range(21):
                        data.append(0)
                        data.append(0)

                # right hand 21
                if results.right_hand_landmarks != None:                     
                    for data_points in results.right_hand_landmarks.landmark:
                        data.append(data_points.x)
                        data.append(data_points.y)
                else:
                    for i in range(21):
                        data.append(0)
                        data.append(0)


                # body 33
                if results.pose_landmarks != None:
                    for data_points in results.pose_landmarks.landmark:
                        data.append(data_points.x)
                        data.append(data_points.y)
                else:
                    for i in range(33):
                        data.append(0)
                        data.append(0)

                writer.writerow(data)
                
                # drawing for testing
                # cv2.imwrite(store_path, image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()