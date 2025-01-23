import json
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class SubfigureDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        PyTorch Dataset class to load images from subfig_path and apply transformations.

        Args:
            data_list (List[Dict]): List of dictionaries with dataset information.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        subfig_path = item["subfig_path"]
        image = Image.open(subfig_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx


class Fig_Separation_Dataset(Dataset):
    def __init__(
        self,
        filepath,
        only_medical=True,
        normalization=False,
        start=0,
        end=-1,
        input_size=512,
    ):
        self.images = []  # list of {'path':'xxx/xxx.png', 'w':256, 'h':256}
        if normalization:
            self.image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.image_transform = transforms.Compose([transforms.ToTensor()])

        # preprocessing
        lines = open(filepath).readlines()
        dataset = [json.loads(line) for line in lines]

        if only_medical:
            dataset = [data for data in dataset if data["is_medical"]]

        dataset = dataset[start : len(dataset)]
        filtered_compound_fig_num = 0
        print(f"Total {len(dataset)} Compound Figures.")
        count = start

        for datum in tqdm(dataset):
            image_info = {}
            image_info["path"] = datum["image_path"]
            image_info["id"] = datum["id"]
            image_info["index"] = count
            image_info["w"] = datum["width"]
            image_info["h"] = datum["height"]
            count += 1

            self.images.append(image_info)
            filtered_compound_fig_num += 1

        self.input_size = input_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        unpadded_image = Image.open(self.images[index]["path"]).convert("RGB")
        unpadded_image = self.image_transform(unpadded_image)

        return (
            unpadded_image,
            self.images[index]["h"],
            self.images[index]["w"],
            self.images[index]["id"],
            self.images[index]["index"],
            self.input_size,
        )


def fig_separation_collate(data):
    """
    Args:
        data: refer to __getitem__() in FigCap_Dataset

    Returns
    -------
        images: tensor (bs, 3, max_h, max_w)
        # subfigs: list of lists  [ ... [box(tensor, (subfig_num, 4)), alignment(tensor, (subfig_num, max_l))], ... ]
        other info: ......
    """
    pad_imgs = []
    unpadded_hws = []
    image_ids = []
    image_index = []
    unpadded_images = []

    for sample in data:
        unpadded_image, unpadded_h, unpadded_w, sample_id, index, input_size = sample
        image_ids.append(sample_id)
        image_index.append(index)
        unpadded_hws.append([unpadded_h, unpadded_w])

        _, h, w = unpadded_image.shape
        scale = min(input_size / h, input_size / w)
        resize_transform = transforms.Resize([round(scale * h), round(scale * w)])
        resized_img = resize_transform(unpadded_image)  # resize within input_size
        pad = (0, input_size - round(scale * w), 0, input_size - round(scale * h))
        padded_img = F.pad(
            resized_img, pad, "constant", 0
        )  # pad image to input_size x input_size
        pad_imgs.append(padded_img)

        unpadded_images.append(unpadded_image)  # [bs * (3, h, w)]

    pad_imgs = torch.stack(pad_imgs, dim=0)  # (bs, 3, max_w, max_h)

    return {
        "image": pad_imgs,
        "unpadded_hws": unpadded_hws,
        "image_id": image_ids,
        "image_index": image_index,
        "original_image": unpadded_images,
    }
