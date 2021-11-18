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

with open(r'C:\Users\Xylon\Desktop\Data-SLR\keypoints_norm.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # read all images in the data folder
    root_dir = r'C:\Users\Xylon\Desktop\Data-SLR\data'

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # data = []
            norm = []
            image_path = os.path.join(subdir, file)

            classname = subdir.split("\\")[-2]
            video_id = subdir.split("\\")[-1]
            # data.append(classname)
            # data.append(video_id)

            norm.append(classname)
            norm.append(video_id)

            name = file.split(".")[0] + '.' + file.split(".")[1]
            # data.append(name)
            norm.append(name)

            print("saving keypoints for " + image_path)

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
                            normalized_data = (normalizedLandmark.x, normalizedLandmark.y)
                            # data.append(pixelCoordinatesLandmark)
                            norm.append(normalized_data)

                writer.writerow(norm)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
