import mmcv
import cv2
import numpy as np
import math


def draw_limbs_2d(img, joints_2d, limb_parents):
   for limb_num in range(len(limb_parents)):
        y1 = joints_2d[limb_num, 0]
        x1 = joints_2d[limb_num, 1]
        y2 = joints_2d[limb_parents[limb_num], 0]
        x2 = joints_2d[limb_parents[limb_num], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        		# if length < 10000 and length > 5:
        deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
         								   (int(length / 2), 3),
         								   int(deg),
         								   0, 360, 1)
        cv2.fillConvexPoly(img, polygon, color=(0,255,255))
           
   return img

                         
limb_parents = [1, 1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15]
limb_parents = np.array(limb_parents)

def visualize(video_path, joints):
    video = mmcv.VideoReader(video_path)
    for i, frame in enumerate(video):

    
             limb_img = draw_limbs_2d(frame, joints[i], limb_parents)
             cv2.imshow('2d', limb_img.astype(np.uint8))
             cv2.waitKey(40)  
      


