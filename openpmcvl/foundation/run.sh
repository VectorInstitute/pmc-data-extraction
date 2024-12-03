# download main section
nohup python -u src/fetch_oa.py --extraction-dir /datasets/PMC-15M/ --volumes 11 > output.txt

# download non-comm and other
# Note: different license types must have different extraction-dir
cd openpmcvl/foundation
source ~/Documents/envs/openpmcvl/bin/activate
# comm license
nohup python -u src/fetch_oa.py --num-retries 5 --extraction-dir /datasets/PMC-15M/comm --license-type comm --volumes 0 > /datasets/PMC-15M/comm/output_v0.txt
# noncomm license
nohup python -u src/fetch_oa.py --num-retries 5 --extraction-dir /datasets/PMC-15M/non_comm --license-type noncomm --volumes 0 > output.txt
nohup python -u src/fetch_oa.py --num-retries 5 --extraction-dir /datasets/PMC-15M/non_comm --license-type noncomm --volumes 1 > /datasets/PMC-15M/non_comm/output_v1.txt
# other license
nohup python -u src/fetch_oa.py --num-retries 5 --extraction-dir /datasets/PMC-15M/other --license-type other --volumes 0 > /datasets/PMC-15M/other/output_v0.txt

# dev
python -u src/fetch_oa.py --num-retries 5 --extraction-dir ./PMC_OA2 --license-type noncomm --volumes 1 2 3
