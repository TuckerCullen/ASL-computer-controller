import cv2
import mediapipe
import csv
import os

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

#  create a CSV file
header = ['gloss', 'video_id', 'frame_id',
          'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP',
          'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP',
          'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP',
          'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP',
          'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
          'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP',
          'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP',
          'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP',
          'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP',
          'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

with open(r'C:\Users\Xylon\Desktop\Data-SLR\keypoints.csv', 'w', encoding='UTF8') as f:
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

            name = file.split(".")[0] + '.' + file.split(".")[1]
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

                writer.writerow(data)

                # cv2.imwrite(store_path, image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
