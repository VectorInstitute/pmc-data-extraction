import codecs
from bs4 import BeautifulSoup
import subprocess
import os
import shutil


def get_img_url(PMC_ID, fig_id):
    img_src_url = "https://pmc.ncbi.nlm.nih.gov/articles/%s/figure/%s/" % (PMC_ID, fig_id)
    file_path = f"/datasets/PMC-15M/temp/{PMC_ID}_{fig_id}"
    subprocess.call(
        [
            "wget",
            "-U",
            "Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0",
            "-nc",
            "-nd",
            "-c",
            "-q",
            "-P",
            file_path,
            img_src_url,
        ]
    )
    # find the actual image url in the xml
    xml_path = os.path.join(file_path, "index.html")
    with codecs.open(xml_path, encoding="utf-8") as f:
        document = f.read()
    soup = BeautifulSoup(document, "lxml")
    img = soup.find(name="img", attrs={"class": "graphic"})
    img_url = img.attrs["src"]
    try:
        shutil.rmtree(file_path)
    except Exception as e:
        print(f"Exception occured while deleting directory {file_path}", e)
    return img_url



if __name__ == "__main__":
    PMC_ID = "PMC11000780"
    fig_id = "fig1"
    get_img_url(PMC_ID, fig_id)