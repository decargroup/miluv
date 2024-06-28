import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import rosbag 
import numpy.random as random
import numpy as np

# from pyuwbcalib.utils import set_plotting_env
# set_plotting_env()
np.random.seed(5)
input_bag = '/home/shalaby/Desktop/datasets/miluv_dataset/main/calib/ifo001/ifo001_calib1_2024-02-12-09-15-18.bag'
# b = bagreader(input_bag)

# df = pd.read_csv(b.message_by_topic("/ifo001/camera/color/image_raw/compressed"))

cv_image = []
t_image = []

bridge = CvBridge()
idx_list = random.randint(0, 200000, 9)
idx_list.sort()
i = 0 
for topic, msg, t in rosbag.Bag(input_bag).read_messages():
    if i == 0:
        t0 = t.to_sec()
    if i < idx_list[0]:
        i += 1
        continue
    if topic != "/ifo001/camera/color/image_raw/compressed":
        continue
    cv_image = cv_image + [bridge.compressed_imgmsg_to_cv2(msg)]
    t_image = t_image + [t.to_sec()]
    idx_list = idx_list[1:]
    
    if len(idx_list) == 0:
        break

# print(t_image)
fig, axs = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(cv_image[i*3+j])
        axs[i, j].axis('off')
        axs[i,j].set_title("t = " + str(np.round(t_image[i*3+j] - t0)) + " s")
plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')

plt.savefig("figs/kalibr_calib.pdf")

plt.show()
