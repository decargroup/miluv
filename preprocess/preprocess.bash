# python preprocess/read_bags.py data/1c False
# python preprocess/process_uwb.py data/1c
# python preprocess/cleanup_csv.py data/1c

python preprocess/read_bags.py data/16 False
python preprocess/process_uwb.py data/16
python preprocess/cleanup_csv.py data/16

# python preprocess/read_bags.py data/20 False
# python preprocess/process_uwb.py data/20
# python preprocess/cleanup_csv.py data/20