from random import choices

import torch
from torchvision import transforms


class Augmentation_Tool():
    def __init__(self, aug_params):
        super().__init__()
        # aug param
        if aug_params:
            # flip
            self.horizontal_flip_prob = aug_params['horizontal_flip_prob']
            self.vertical_flip_prob = aug_params['vertical_flip_prob']
            # color jitter
            self.colorjitter_prob = aug_params['colorjitter_prob']
            self.brightness = aug_params['brightness']
            self.contrast = aug_params['contrast']
            self.saturation = aug_params['saturation']
            self.color_jitter = transforms.ColorJitter(self.brightness, self.contrast, self.saturation)
            # noise and blur
            self.gaussian_noise_prob = aug_params['gaussian_noise_prob']
            self.gaussian_blur_prob = aug_params['gaussian_blur_prob']
            self.gaussian_blur = transforms.GaussianBlur((5,5))
            # gray scale and color invert
            self.gray_scale_prob = aug_params['gray_scale_prob']
            self.gray_converter = transforms.Grayscale(3)
            self.color_inverter = transforms.RandomInvert(aug_params['color_invert_prob'])
        else:
            self.horizontal_flip_prob = self.vertical_flip_prob = 0.0
            self.colorjitter_prob = 0.0 
            self.color_jitter = None
            self.gaussian_noise_prob = self.gaussian_blur_prob = 0.0
            self.gaussian_blur = None
            self.gray_scale_prob = 0.0
            self.gray_converter = None
            self.color_inverter = transforms.RandomInvert(0.0)
            
    def __call__(self, unpadded_image, unnorm_bboxes):
        unpadded_image, unnorm_bboxes = self.apply_horizontal_flip(unpadded_image, unnorm_bboxes)
        unpadded_image, unnorm_bboxes = self.apply_vertical_flip(unpadded_image, unnorm_bboxes)
        unpadded_image = self.apply_color_jit(unpadded_image)
        unpadded_image = self.color_inverter(unpadded_image)
        unpadded_image = self.apply_gray_scale(unpadded_image)
        unpadded_image = self.apply_gaussian_noise(unpadded_image)
        unpadded_image = self.apply_gaussian_blur(unpadded_image)
        return unpadded_image, unnorm_bboxes

    def apply_horizontal_flip(self, image, bboxes):
        """
        此Aug函数应在getitem函数中调用

        Args:
            image (tensor): (3, h, w)
            bboxes (list of [cx, cy, w, h]): unnormalized
        """
        if choices([0, 1], weights=[1-self.horizontal_flip_prob, self.horizontal_flip_prob])[0]:
            image = transforms.RandomHorizontalFlip(p=1)(image)
            for bbox in bboxes: # cx, cy, w, h
                bbox[0] = image.shape[-1]-bbox[0]
        return image, bboxes

    def apply_vertical_flip(self, image, bboxes):
        """
        此Aug函数应在getitem函数中调用

        Args:
            image (tensor): 
            bboxes (list of [cx, cy, w, h]): unnormalized
        """
        if choices([0, 1], weights=[1-self.horizontal_flip_prob, self.horizontal_flip_prob])[0]:
            image = transforms.RandomVerticalFlip(p=1)(image)
            for bbox in bboxes: # cx, cy, w, h
                bbox[1] = image.shape[1]-bbox[1]
        return image, bboxes

    def apply_color_jit(self, image):
        """
        此Aug函数应在getitem函数中调用

        Args:
            image (tensor): (3, h, w)
        """
        if choices([0, 1], weights=[1-self.colorjitter_prob, self.colorjitter_prob])[0]:
            image = self.color_jitter(image)
        return image

    def apply_gaussian_blur(self, image):
        """
        此Aug函数应在getitem函数中调用

        Args:
            image (tensor): (3, h, w)
        """
        if choices([0, 1], weights=[1-self.gaussian_blur_prob, self.gaussian_blur_prob])[0]:
            image = self.gaussian_blur(image)
        return image

    def apply_gray_scale(self, image):
        """
        此Aug函数应在getitem函数中调用

        Args:
            image (tensor): (3, h, w)
        """
        if choices([0, 1], weights=[1-self.gray_scale_prob, self.gray_scale_prob])[0]:
            image = self.gray_converter(image)
        return image

    def apply_gaussian_noise(self, image, mean=0, std=0.1):
        """
        此Aug函数应在getitem函数中调用

        Args:
            image (tensor): (3, h, w)
        """
        if choices([0, 1], weights=[1-self.gaussian_noise_prob, self.gaussian_noise_prob])[0]:
            noise = torch.normal(mean,std,image.shape)
            image = image+noise
        return image