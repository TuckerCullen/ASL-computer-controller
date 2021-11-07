import cv2
import os

# scan all raw mp4 videos e.g. raw/open/1.mp4, raw/close/1.mp4
root_dir = r'C:\Users\Xylon\Desktop\WLASL'
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        video_path = os.path.join(subdir, file)

        # construct store path
        try:
            if not os.path.exists('data'):
                os.makedirs('data')
        except OSError:
            print ('Error: Creating directory of data')

        store_dir = r'C:\Users\Xylon\Desktop\SLR\mediapipe\data'
        parents = subdir.split("\\")
        dir_parent = parents[-1]
        store_parent_dir = os.path.join(store_dir, dir_parent)
        try:
            if not os.path.exists(store_parent_dir):
                os.makedirs(store_parent_dir)
        except OSError:
            print('Error: Creating directory')

        name = file.split(".")[0]
        store_path = os.path.join(store_parent_dir, name)
        try:
            if not os.path.exists(store_path):
                os.makedirs(store_path)
        except OSError:
            print('Error: Creating directory')

        # Playing video from file:
        cap = cv2.VideoCapture(video_path)

        # capture frames and save
        count = 0
        while cap.isOpened():
            ret, image = cap.read()

            if ret and count <= 100:            # count maximum 100 (few videos cannot stop)
                name = os.path.join(store_path, str(count) + '.jpg')
                cv2.imwrite(name, image)
                count += 5      # frame rate
                cap.set(1, count)
                print('Creating...' + name)
            else:
                cap.release()
                break

        # When everything done, close all windows
        cv2.destroyAllWindows()
