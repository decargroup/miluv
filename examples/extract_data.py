from miluv.data import DataLoader
import numpy as np
import cv2

mv_3a = DataLoader(
    "3a",
    cir=False,
    barometer=False,
    height=False,
)

# Time in sec
timestamps = np.arange(0, 100, 1)

# We are leaving out ifo003
robots = ["ifo001", "ifo002"]

# Fetching just the imu data
cams = ["bottom", "color"]

imgs_at_timestamps = mv_3a.imgs_from_timestamps(timestamps, robots, cams)

for image in imgs_at_timestamps["ifo001"]["bottom"]["image"]:
    # show the output image after AprilTag detection
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)
    #
    # press the escape key to break from the loop
    if key == 27:
        break
