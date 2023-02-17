from pathlib import Path
import json
import numpy as np


def read_OpenPose(dir_path,
              frame_width,
              frame_height,
              label='unknown',
              label_index=-1):
    sequence_info = []
    p = Path(dir_path)
    for path in p.glob('*.json'):
        json_path = str(path)
        frame_id = int(path.stem.split('_')[-2])
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            skeleton = {}
            keypoints = person['pose_keypoints_2d']
            for i in range(0, len(keypoints), 3):
                coordinates += [
                    keypoints[i] / frame_width, keypoints[i + 1] / frame_height
                ]
                score += [keypoints[i + 2]]
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index

    return video_info


def read_OpenPose_vis(dir_path,
              label='unknown',
              label_index=-1):
    sequence_info = []
    p = Path(dir_path)
    for path in p.glob('*.json'):
        json_path = str(path)
        frame_id = int(path.stem.split('_')[-2])
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            skeleton = {}
            keypoints = person['pose_keypoints_2d']
            for i in range(0, len(keypoints), 3):
                coordinates += [
                    keypoints[i], keypoints[i + 1]
                ]
                score += [keypoints[i + 2]]
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index

    return video_info


def OpenPose_convert(path, w, h): 
    data = read_OpenPose(path,  w, h)
    
    X = []
    for i in range(0, len(data['data'])):
       
        coords = data['data'][i]['skeleton'][0]['pose']
        conf = data['data'][i]['skeleton'][0]['score']
        coords = np.array(coords).reshape([18, 2])
        x = np.concatenate((coords, np.array(conf).reshape([18, 1])), axis = 1).astype('float32')
        X.append(x)
    X = np.array(X)
    X = X.reshape([X.shape[2], X.shape[0], X.shape[1], 1])
    Z = np.zeros(X.shape)
    X = np.concatenate((X,Z), axis = 3)
    return X.astype('float32')
    
def OpenPose_convert_vis(path): 
    data = read_OpenPose_vis(path)
    
    X = []
    for i in range(0, len(data['data'])):
       
        coords = data['data'][i]['skeleton'][0]['pose']
        coords = np.array(coords).reshape([18, 2])
        X.append(coords)
    X = np.array(X).astype('float32')

    return X

