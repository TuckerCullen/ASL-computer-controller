import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# create encoding dictionary
def labelDict(df):
    gloss_dict = {}
    count = 0
    for gloss in df.gloss:
        if gloss not in gloss_dict.keys():
            gloss_dict[gloss] = count
            count += 1
    return gloss_dict

# process data points
def getKeypoints(df):   
    data = []
    for i in range(len(df)):
        frame = []
        for col in range(3, 45):
            keypoint = df.iloc[i, col]
            if keypoint == "(0,0)":
                frame.append(-1)
                frame.append(-1)
            else:
                keypoint = keypoint.strip("(")
                keypoint = keypoint.strip(')')
                coor = keypoint.split(", ")
                frame.append(int(coor[0]))
                frame.append(int(coor[1]))
        frame = np.array(frame, dtype=np.int32)
        data.append(frame)    
    return data

def get_train_test(df):                   # df = pd.read_csv("processed.csv")
    df = df.drop(columns='Unnamed: 0')
    
    # process labels
    gloss_dict = labelDict(df)
    label = []
    gloss = df.gloss
    for i in range(int(len(gloss) / 70)):
        label.append(gloss_dict[gloss[i * 70]])
    y = np.array(label)
    LABELS = [key for key in gloss_dict.keys()]

    X = np.array(getKeypoints(df))   # len(X) == 1284850
    X_ = np.array(X)      # X_.shape == (1284850, 84)
    blocks = int(len(X) / 70)
    X_ = np.array(np.split(X_,blocks))      # X_.shape == (18355, 70, 84) 18355 valid videos, 70 time frames, 84 features

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, LABELS