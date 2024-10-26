import numpy as np
import cv2
import matplotlib.pyplot as plt

# Some global variables for draw_rectangle_drag function
draging = False     # True while dragind
start_x = -1
start_y = -1

# This function takes mouse parameters from setMousecallback and drw our circle
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(blank_image1,(x,y),70,(0,255,0),-1)

    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(blank_image1, (x, y), 70, (0, 0, 255), -1)



def draw_rectangle_drag(event,x,y,flags,param):
    global draging, start_x, start_y

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        draging = True

    elif event == cv2.EVENT_MOUSEMOVE and draging:
        cv2.rectangle(blank_image2 ,(start_x, start_y), (x, y), (0, 255, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(blank_image2, (start_x, start_y), (x, y), (0, 255, 0), -1)
        draging = False


# Build a window and name it
cv2.namedWindow('Blank Image1')

# Build another window
cv2.namedWindow('Blank Image2')

# This function detect Mouse action on our window and call draw_circle function which we made
# Note that: this function passes some mouse parameters like event and (x,y) to our function
cv2.setMouseCallback('Blank Image1', draw_circle)

cv2.setMouseCallback('Blank Image2', draw_rectangle_drag)



blank_image1 = np.zeros((512, 512, 3), np.uint8)

blank_image2 = np.zeros((512, 512, 3), np.uint8)

while True:
    cv2.imshow('Blank Image1', blank_image1)

    cv2.imshow('Blank Image2', blank_image2)
    # wait 1ms AND if you pressed Esc key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
