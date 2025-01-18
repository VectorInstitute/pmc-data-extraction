import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from skimage import io
from torch.autograd import Variable

from openpmcvl.granular.models.network import resnet152
from openpmcvl.granular.models.process import postprocess, preprocess, yolobox2label
from openpmcvl.granular.models.yolov3 import YOLOv3


class classifier:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        configuration_file = os.path.join(
            self.current_dir, "..", "config", "yolov3_default_subfig.cfg"
        )

        with open(configuration_file, "r") as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)

        self.image_size = configuration["TEST"]["IMGSIZE"]
        self.nms_threshold = configuration["TEST"]["NMSTHRE"]
        self.confidence_threshold = 0.0001
        self.dtype = torch.cuda.FloatTensor
        self.device = torch.device("cuda")

        object_detection_model = YOLOv3(configuration["MODEL"])
        self.object_detection_model = self.load_model_from_checkpoint(
            object_detection_model, "object_detection_model.pt"
        )
        ## Load text recognition model
        text_recognition_model = resnet152()
        self.text_recognition_model = self.load_model_from_checkpoint(
            text_recognition_model, "text_recognition_model.pt"
        )

        self.object_detection_model.eval()
        self.text_recognition_model.eval()

    def load_model_from_checkpoint(self, model, model_name):
        """Load checkpoint weights into model"""
        checkpoints_path = os.path.join(self.current_dir, "..", "checkpoints")
        checkpoint = os.path.join(checkpoints_path, model_name)
        model.load_state_dict(torch.load(checkpoint))
        model.to(self.device)
        return model

    def detect_subfigure_boundaries(self, figure_path):
        """Detects the bounding boxes of subfigures in figure_path

        Args:
            figure_path: A string, path to an image of a figure
                from a scientific journal
        Returns:
            subfigure_info (list of lists): Each inner list is
                x1, y1, x2, y2, confidence
        """
        ## Preprocess the figure for the models
        img = io.imread(figure_path)
        if len(np.shape(img)) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        img, info_img = preprocess(img, self.image_size, jitter=0)
        img = np.transpose(img / 255.0, (2, 0, 1))
        img = np.copy(img)
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = Variable(img.type(self.dtype))

        img_raw = Image.open(figure_path).convert("RGB")
        width, height = img_raw.size

        ## Run model on figure
        with torch.no_grad():
            outputs = self.object_detection_model(img.to(self.device))
            outputs = postprocess(
                outputs,
                dtype=self.dtype,
                conf_thre=self.confidence_threshold,
                nms_thre=self.nms_threshold,
            )

        ## Reformat model outputs to display bounding boxes in our desired format
        ## List of lists where each inner list is [x1, y1, x2, y2, confidence]
        subfigure_info = list()

        if outputs[0] is None:
            return subfigure_info

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
            box = yolobox2label(
                [
                    y1.data.cpu().numpy(),
                    x1.data.cpu().numpy(),
                    y2.data.cpu().numpy(),
                    x2.data.cpu().numpy(),
                ],
                info_img,
            )
            box[0] = int(min(max(box[0], 0), width - 1))
            box[1] = int(min(max(box[1], 0), height - 1))
            box[2] = int(min(max(box[2], 0), width))
            box[3] = int(min(max(box[3], 0), height))
            # ensures no extremely small (likely incorrect) boxes are counted
            small_box_threshold = 5
            if (
                box[2] - box[0] > small_box_threshold
                and box[3] - box[1] > small_box_threshold
            ):
                box.append("%.3f" % (cls_conf.item()))
                subfigure_info.append(box)
        return subfigure_info

    def detect_subfigure_labels(self, figure_path, subfigure_info):
        """Uses text recognition to read subfigure labels from figure_path

        Note:
            To get sensible results, should be run only after
            detect_subfigure_boundaries has been run
        Args:
            figure_path (str): A path to the image (.png, .jpg, or .gif)
                file containing the article figure
            subfigure_info (list of lists): Details about bounding boxes
                of each subfigure from detect_subfigure_boundaries(). Each
                inner list has format [x1, y1, x2, y2, confidence] where
                x1, y1 are upper left bounding box coordinates as ints,
                x2, y2, are lower right, and confidence the models confidence
        Returns:
            subfigure_info (list of tuples): Details about bounding boxes and
                labels of each subfigure in figure. Tuples for each subfigure are
                (x1, y1, x2, y2, label) where x1, y1 are upper left x and y coord
                divided by image width/height and label is the an integer n
                meaning the label is the nth letter
            concate_img (np.ndarray): A numpy array representing the figure.
                Used in classify_subfigures. Ideally this will be removed to
                increase modularity.
        """
        img_raw = Image.open(figure_path).convert("RGB")
        img_raw = img_raw.copy()
        width, height = img_raw.size
        binary_img = np.zeros((height, width, 1))

        detected_label_and_bbox = None
        max_confidence = 0.0
        for subfigure in subfigure_info:
            ## Preprocess the image for the model
            bbox = tuple(subfigure[:4])
            img_patch = img_raw.crop(bbox)
            img_patch = np.array(img_patch)[:, :, ::-1]
            img_patch, _ = preprocess(img_patch, 28, jitter=0)
            img_patch = np.transpose(img_patch / 255.0, (2, 0, 1))
            img_patch = torch.from_numpy(img_patch).type(self.dtype).unsqueeze(0)

            ## Run model on figure
            label_prediction = self.text_recognition_model(img_patch.to(self.device))
            label_confidence = np.amax(
                F.softmax(label_prediction, dim=1).data.cpu().numpy()
            )
            x1, y1, x2, y2, box_confidence = subfigure
            total_confidence = float(box_confidence) * label_confidence
            if total_confidence < max_confidence:
                continue
            label_value = chr(
                label_prediction.argmax(dim=1).data.cpu().numpy()[0] + ord("a")
            )
            if label_value == "z":
                continue
            detected_label_and_bbox = [label_value, x1, y1, x2, y2]

        return detected_label_and_bbox

    def run(self, figure_path):
        subfigure_info = self.detect_subfigure_boundaries(figure_path)
        subfigure_info = self.detect_subfigure_labels(figure_path, subfigure_info)

        return subfigure_info
