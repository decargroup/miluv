import numpy as np
from bagpy import bagreader
import matplotlib.pyplot as plt
import pandas as pd

from pyuwbcalib.utils import set_plotting_env

set_plotting_env()

b = bagreader('/home/shalaby/Desktop/datasets/miluv_dataset/main/12/ifo001_exp12c_2024-02-01-13-21-48.bag')

df = pd.read_csv(b.message_by_topic("/ifo001/uwb/cir"))
cir_cols = df.columns[df.columns.str.contains('cir')]
df["cir"] = df[cir_cols].values.tolist()
df = df.drop(columns=cir_cols)

k = 105
plt.plot(df.iloc[k].cir, label="CIR")
plt.plot(np.ones(2)*df.iloc[k].idx, [0,8000], linewidth=3, label="Peak")
plt.ylabel("Amplitude ")
plt.xlabel("Sample index")
plt.xlim([675, 900])
plt.ylim([0, 7000])
plt.legend()

plt.savefig("figs/cir_example.pdf")
