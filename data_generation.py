'''This script is for generating data
1. Provide desired path to store images.
2. Press 'c' to capture image and display it.
3. Press any button to continue.
4. Press 'q' to quit.
'''

import cv2
import pyrealsense2 as rs
import numpy as np
path = "/home/sung/workspace/object_detection/explore/aruco_data/"
count = 0
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    cv2.imshow("img", color_image)


    if cv2.waitKey(20) & 0xFF == ord('c'):
        cv2.imwrite(name, img)
        cv2.imshow("img", img)
        count += 1
        if cv2.waitKey(0) & 0xFF == ord('q'):

            break;
