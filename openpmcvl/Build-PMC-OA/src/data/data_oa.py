'''File list for PMC OA: [csv, txt, tar]
'''

MAX_VOLUME_NUM = 12
UPDATE_SCHEDULE = '2024-06-18'
volume = f'PMC0%02dxxxxxx'
csv_url = f'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/oa_comm_xml.PMC0%02dxxxxxx.baseline.{UPDATE_SCHEDULE}.filelist.csv'
txt_url = f'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/oa_comm_xml.PMC0%02dxxxxxx.baseline.{UPDATE_SCHEDULE}.filelist.txt'
tar_url = f'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/oa_comm_xml.PMC0%02dxxxxxx.baseline.{UPDATE_SCHEDULE}.tar.gz'

OA_LINKS = {}

for volume_id in range(MAX_VOLUME_NUM):
    OA_LINKS[volume % volume_id] = {}
    OA_LINKS[volume % volume_id]['csv_url'] = csv_url % volume_id
    OA_LINKS[volume % volume_id]['txt_url'] = txt_url % volume_id
    OA_LINKS[volume % volume_id]['tar_url'] = tar_url % volume_id


if __name__ == '__main__':
    print(OA_LINKS)
