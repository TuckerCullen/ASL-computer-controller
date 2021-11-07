import cv2
import mediapipe
import csv
import os

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

#  create a CSV file
header = ['gloss', 'video_id', 'frame_id', 'WRIST_p', 'WRIST_n', 'THUMB_CMC_p', 'THUMB_CMC_n',
          'THUMB_MCP_p', 'THUMB_MCP_n', 'THUMB_IP_p', 'THUMB_IP_n',
          'THUMB_TIP_p', 'THUMB_TIP_n', 'INDEX_FINGER_MCP_p', 'INDEX_FINGER_MCP_n',
          'INDEX_FINGER_PIP_p', 'INDEX_FINGER_PIP_n', 'INDEX_FINGER_DIP_p', 'INDEX_FINGER_DIP_n',
          'INDEX_FINGER_TIP_p','INDEX_FINGER_TIP_n', 'MIDDLE_FINGER_MCP_p', 'MIDDLE_FINGER_MCP_n',
          'MIDDLE_FINGER_PIP_p', 'MIDDLE_FINGER_PIP_n', 'MIDDLE_FINGER_DIP_p', 'MIDDLE_FINGER_DIP_n',
          'MIDDLE_FINGER_TIP_p', 'MIDDLE_FINGER_TIP_n', 'RING_FINGER_MCP_p', 'RING_FINGER_MCP_n',
          'RING_FINGER_PIP_p', 'RING_FINGER_PIP_n', 'RING_FINGER_DIP_p', 'RING_FINGER_DIP_n',
          'RING_FINGER_TIP_p', 'RING_FINGER_TIP_n', 'PINKY_MCP_p', 'PINKY_MCP_n',
          'PINKY_PIP_p', 'PINKY_PIP_n', 'PINKY_DIP_p', 'PINKY_DIP_n',
          'PINKY_TIP_p', 'PINKY_TIP_n',
          'WRIST_p', 'WRIST_n', 'THUMB_CMC_p', 'THUMB_CMC_n',
          'THUMB_MCP_p', 'THUMB_MCP_n', 'THUMB_IP_p', 'THUMB_IP_n',
          'THUMB_TIP_p', 'THUMB_TIP_n', 'INDEX_FINGER_MCP_p', 'INDEX_FINGER_MCP_n',
          'INDEX_FINGER_PIP_p', 'INDEX_FINGER_PIP_n', 'INDEX_FINGER_DIP_p', 'INDEX_FINGER_DIP_n',
          'INDEX_FINGER_TIP_p','INDEX_FINGER_TIP_n', 'MIDDLE_FINGER_MCP_p', 'MIDDLE_FINGER_MCP_n',
          'MIDDLE_FINGER_PIP_p', 'MIDDLE_FINGER_PIP_n', 'MIDDLE_FINGER_DIP_p', 'MIDDLE_FINGER_DIP_n',
          'MIDDLE_FINGER_TIP_p', 'MIDDLE_FINGER_TIP_n', 'RING_FINGER_MCP_p', 'RING_FINGER_MCP_n',
          'RING_FINGER_PIP_p', 'RING_FINGER_PIP_n', 'RING_FINGER_DIP_p', 'RING_FINGER_DIP_n',
          'RING_FINGER_TIP_p', 'RING_FINGER_TIP_n', 'PINKY_MCP_p', 'PINKY_MCP_n',
          'PINKY_PIP_p', 'PINKY_PIP_n', 'PINKY_DIP_p', 'PINKY_DIP_n',
          'PINKY_TIP_p', 'PINKY_TIP_n']

with open('keypoints.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # read all images in the data folder
    root_dir = r'C:\Users\Xylon\Desktop\Data-SLR\data'
    des_dir = r'C:\Users\Xylon\Desktop\Data-SLR\keypoints'
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

            name = file.split(".")[0]
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

            with handsModule.Hands(static_image_mode=True) as hands:
                image = cv2.imread(image_path)

                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                imageHeight, imageWidth, _ = image.shape

                if results.multi_hand_landmarks != None:
                    for handLandmarks in results.multi_hand_landmarks:
                        drawingModule.draw_landmarks(image, handLandmarks, handsModule.HAND_CONNECTIONS)

                        for point in handsModule.HandLandmark:
                            normalizedLandmark = handLandmarks.landmark[point]
                            pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                                        normalizedLandmark.y,
                                                                                                      imageWidth, imageHeight)

                            # print(point)
                            # print(pixelCoordinatesLandmark)
                            # print(normalizedLandmark)
                            data.append(pixelCoordinatesLandmark)
                            normalized = (normalizedLandmark.x, normalizedLandmark.y, normalizedLandmark.z)
                            data.append(normalized)

                # writer.writerow(data)

                # cv2.imshow('Test image', image)

                # cv2.imwrite(store_path, image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
