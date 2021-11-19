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

def get_data(df):                   # df = pd.read_csv("processed.csv")
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
    
    np.savetxt('X.txt', X_, fmt='%d')
    np.savetxt('y.txt', y, fmt='%d')
    np.savetxt('labels.txt', LABELS, fmt='%s')