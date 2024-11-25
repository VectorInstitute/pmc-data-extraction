# download main section
nohup python -u src/fetch_oa.py --extraction-dir /datasets/PMC-15M/ --volumes 11 > output.txt

# download non-comm and other
cd openpmcvl/foundation
source ~/Documents/envs/openpmcvl/bin/activate
nohup python -u src/fetch_oa.py --extraction-dir /datasets/PMC-15M/non_comm --volumes 0 > output.txt
nohup python -u src/fetch_oa.py --extraction-dir /datasets/PMC-15M/non_comm --volumes 1 > /datasets/PMC-15M/non_comm/output_v1.txt

# dev
python -u src/fetch_oa.py --volumes 0