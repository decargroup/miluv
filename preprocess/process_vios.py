import sys
from os import listdir, remove, walk, rename
from os.path import join, dirname, basename
import pandas as pd

vio = [
    "timestamp",
    "position.x",
    "position.y",
    "position.z",
    "orientation.x",
    "orientation.y",
    "orientation.z",
    "orientation.w",
    "velocity.x",
    "velocity.y",
    "velocity.z",
]

vio_loop = [
    "timestamp",
    "position.x",
    "position.y",
    "position.z",
    "orientation.x",
    "orientation.y",
    "orientation.z",
    "orientation.w",
]

def cleanup_vios(dir, path, exp):
    # Find all csv files
    files = [f for f in listdir(dir) if f.endswith('.csv') if exp in f]

    for file in files:
        # if file == 'ifo001_vio.csv':
        if "vio" in file and "loop" not in file:
            process_vio(dir, path, exp, file, vio, "vio")
        elif "loop" in file and "aligned" not in file:
            process_vio(dir, path, exp, file, vio_loop, "vio_loop")


def process_vio(dir, path, exp, file, headers, name):
    df = pd.read_csv(join(dir, file))
    if name == "vio" or name == "vio_loop":
        df = df.iloc[:, :-1]
        df.columns = headers
        df['timestamp'] = df['timestamp'] / 1e9
    path = join(path, exp)
    df.to_csv(join(path, name + ".csv"), index=False)

if __name__ == '__main__':

    # if len(sys.argv) != 2:
    #     print(
    #         "Not enough arguments. Usage: python process_vios.py path_to_csvs")
    #     sys.exit(1)

    # path = sys.argv[1]
    path = "/media/syedshabbir/Seagate B/data/1c"
    vio_path = join(dirname(path), 'vio', basename(path))
    files = [f for f in listdir(path) if f.endswith('.bag')]

    for file in files:

        cleanup_vios(dir = vio_path, 
                     path = path, 
                     exp = file.split(".")[0])