nohup python -u src/fetch_oa.py --extraction-dir /datasets/PMC-15M/ --volumes 11 > output.txt


cd openpmcvl/foundation
source ~/Documents/envs/openpmcvl/bin/activate
nohup python -u src/fetch_oa.py --extraction-dir /datasets/PMC-15M/non_comm --volumes 0 > output.txt
nohup python -u src/fetch_oa.py --extraction-dir /datasets/PMC-15M/non_comm --volumes 1 > /datasets/PMC-15M/non_comm/output_v1.txt