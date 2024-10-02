import pyrealsense2 as rs
import numpy as np
import cv2
import time

# initialize the camera pipeline
pipeline = rs.pipeline()
config = rs.config()

# enalbe depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# start the camera streaming
pipeline.start(config)

try:
    # while True:
    for i in range(100):
        # wait for a frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # expand the depth image to 3 channels
        depth_3d = np.dstack((depth_image, depth_image, depth_image))

        # combine the color and depth images
        combined_image = np.dstack((color_image, depth_3d[..., 0]))

        # generate file name
        filename = f"realsense/imgsrc/{i:03d}.npy"

        # save the combined image as a numpy array
        np.save(filename, combined_image)
        print(f"successfully saved: {filename}")
        time.sleep(1)

finally:
    # stop the camera pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
