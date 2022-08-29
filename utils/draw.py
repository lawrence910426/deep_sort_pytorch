import numpy as np
import cv2
from utils.shapes import Box, Line

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def draw_flow(img, flow):
    keys = list(flow.keys())
    for i in range(len(keys)):
        k = keys[i]
        label = str(k) + ": " + str(flow[k])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 ,2)[0]
        cv2.putText(
            img, label, (10, 10 + (t_size[1] + 3) * (i + 1)), 
            cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2
        )
    return img

foreground = cv2.imread('counter/detector.png', cv2.IMREAD_UNCHANGED)
def draw_detector(background, detector: Line):
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = np.ones((background.shape[0], background.shape[1]))
    alpha_foreground = foreground[:,:,3] / 255.0

    # set adjusted colors
    for i in range(foreground.shape[0]):
        for j in range(foreground.shape[1]):
            for col in range(3):
                x = detector.x1 + i / foreground.shape[0] * detector.x2
                y = detector.y1 + j / foreground.shape[1] * detector.y2
                background[x, y, col] = alpha_foreground[i, j] * foreground[i, j, col] + \
                    alpha_background * background[x, y, col] * (1 - alpha_foreground[i, j])     
    
    return background

if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
