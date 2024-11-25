"""Define FTP paths where Pubmed article lists and article texts are stored."""

MAX_VOLUME_NUM = 12
UPDATE_SCHEDULE = "2024-06-18"
volume = "PMC0%02dxxxxxx"

OA_LINKS = {}

for license_type in ["comm", "noncomm", "other"]:
    csv_url = f"https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_{license_type}/xml/oa_{license_type}_xml.PMC0%02dxxxxxx.baseline.{UPDATE_SCHEDULE}.filelist.csv"
    txt_url = f"https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_{license_type}/xml/oa_{license_type}_xml.PMC0%02dxxxxxx.baseline.{UPDATE_SCHEDULE}.filelist.txt"
    tar_url = f"https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_{license_type}/xml/oa_{license_type}_xml.PMC0%02dxxxxxx.baseline.{UPDATE_SCHEDULE}.tar.gz"

    OA_LINKS[license_type] = {}

    for volume_id in range(MAX_VOLUME_NUM):
        OA_LINKS[license_type][volume % volume_id] = {}
        OA_LINKS[license_type][volume % volume_id]["csv_url"] = csv_url % volume_id
        OA_LINKS[license_type][volume % volume_id]["txt_url"] = txt_url % volume_id
        OA_LINKS[license_type][volume % volume_id]["tar_url"] = tar_url % volume_id
