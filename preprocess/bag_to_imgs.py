import sys
from os import listdir, mkdir
from os.path import join
import rospy
import rosbag
import cv2
from cv_bridge import CvBridge

def write_imgs(input_bag, dir_main):
    bridge = CvBridge()
    for topic, msg, t in rosbag.Bag(input_bag).read_messages():
        if rospy.is_shutdown():
            break
        if msg._type == 'sensor_msgs/CompressedImage':
            if "infra1" in topic:
                dir = dir_main + "infra1/"
            elif "infra2" in topic:
                dir = dir_main + "infra2/"
            elif "bottom" in topic:
                dir = dir_main + "bottom/"
            else:
                continue
            try:
                cv_image = bridge.compressed_imgmsg_to_cv2(msg)
                cv2.imwrite(dir + str(msg.header.stamp) + '.jpeg', cv_image)
            except Exception as e:
                rospy.logwarn('Error uncompressing image: {}'.format(e))
        else:
            continue

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Not enough arguments. Usage: python bag_to_imgs.py input_bag")
        sys.exit(1)
    
    path = sys.argv[1]
    # output_bag = sys.argv[2]

    files = [f for f in listdir(path) if f.endswith('.bag')]

    rospy.init_node('image_uncompress_node')
    
    for file in files:
        mkdir(join(path, file.split(".")[0]))
        mkdir(join(path, file.split(".")[0] + "/infra1"))
        mkdir(join(path, file.split(".")[0] + "/infra2"))
        mkdir(join(path, file.split(".")[0] + "/bottom"))
        write_imgs(join(path, file), join(path, file.split(".")[0]) + "/")
    