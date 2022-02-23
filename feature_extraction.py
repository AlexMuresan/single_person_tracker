import cv2
import torch
import logging
import networks
import webcolors
import torchvision
import numpy as np

from collections import OrderedDict
from scipy.cluster.vq import kmeans
from utils import load_model, get_affine_transform, transform_logits, convert_rgb_to_names


class CustomModel():
    def __init__(self, model_path, dataset_name, device):
        dataset_settings = {
            'lip': {
                'input_size': [473, 473],
                'num_classes': 20,
                'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes',
                          'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face',
                          'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe',
                          'Right-shoe']
            },
            'atr': {
                'input_size': [512, 512],
                'num_classes': 18,
                'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt',
                          'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg',
                          'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
            },
            'pascal': {
                'input_size': [512, 512],
                'num_classes': 7,
                'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs',
                          'Lower Legs'],
            }
        }
        self.num_classes = dataset_settings[dataset_name]['num_classes']
        self.input_size = dataset_settings[dataset_name]['input_size']

        self.model = networks.init_model('resnet101', num_classes=self.num_classes, pretrained=None)
        state_dict = torch.load(model_path)['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(device)
        self.model.eval()

    def get_model(self):
        return self.model

    def get_input_size(self):
        return self.input_size


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeatureExtractor():
    def __init__(self, load_human_segmentation_model=False, load_human_parsing_model=False,
                 device=None) -> None:
        self.logger = logging.getLogger(__name__)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=15, edgeThreshold=48, patchSize=48)

        self.load_human_segmentation_model = load_human_segmentation_model
        self.load_human_parsing_model = load_human_parsing_model

        if (load_human_segmentation_model or load_human_parsing_model) and (device is None):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device:
            self.device = device

        if load_human_parsing_model:
            self.human_parsing_model = CustomModel('human_parsing_checkpoints/exp-schp-201908261155-lip.pth',
                                                   'lip', self.device).get_model()
            self.input_size = CustomModel('human_parsing_checkpoints/exp-schp-201908261155-lip.pth',
                                          'lip', self.device).get_input_size()
            self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]
            self.human_parsing_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.406, 0.456, 0.485],
                    std=[0.225, 0.224, 0.229])
            ])
            self.logger.info('Loaded human parsing model')

        if load_human_segmentation_model:
            self.human_segmentation_model = load_model(self.device)
            self.human_segmentation_model.aux_classifier = Identity()
            self.human_segmentation_model.classifier = Identity()

            self.imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
            self.preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                              torchvision.transforms.Normalize(
                                                                  mean=self.imagenet_stats[0],
                                                                  std=self.imagenet_stats[1]),
                                                              torchvision.transforms.Resize((256, 128))])

            self.logger.info('Loaded human segmentation model')

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def color_histogram(self, image, use_torch=False):
        bgr_histograms = []

        if use_torch:
            if not torch.is_tensor(image):
                self.logger.info('Image is not Tensor. Converting to tensor')
                image = torch.Tensor(image).to(self.device)
            channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
            for channel in channels:
                bgr_histograms.append(torch.histc(channel, bins=256, min=0, max=256))
        else:
            channels = cv2.split(image)
            for channel in channels:
                bgr_histograms.append(cv2.calcHist([channel], [0], None, [250], [1, 250]))

        return bgr_histograms

    def nnet_features(self, image, cpu_output=False):
        if self.load_human_segmentation_model is False:
            self.logger.error('Neural Network not loaded for feature extraction')
            raise Exception('Neural Network not loaded for feature extraction')

        tensor_image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.human_segmentation_model(tensor_image)['out']

        if cpu_output:
            features = features.cpu().detach()

        return features

    def get_human_parsing_logits(self, image, use_gpu=False):
        if self.load_human_parsing_model is False:
            self.logger.error('Neural Network not loaded for feature extraction')
            raise Exception('Neural Network not loaded for feature extraction')

        h, w, _ = image.shape
        person_center, scale = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, scale, r, self.input_size)
        input = cv2.warpAffine(image, trans, (int(self.input_size[1]), int(self.input_size[0])),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))

        input = self.human_parsing_transform(input)

        if use_gpu:
            input = input.to(self.device)

        with torch.no_grad():
            output = self.human_parsing_model(input.unsqueeze(0).to(self.device))

        upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

        logits_result = transform_logits(upsample_output.data.cpu().numpy(), person_center, scale,
                                         w, h, input_size=self.input_size)

        parsing_result = np.argmax(logits_result, axis=2)

        return parsing_result

    def get_part_mask(self, logits, idx):
        mask = np.zeros((logits.shape[0], logits.shape[1]), dtype=np.uint8)
        mask_idx = np.where(logits == idx)

        for x, y in zip(mask_idx[0], mask_idx[1]):
            mask[x][y] = 1

        return mask

    def get_masked_limbs(self, img, logits, idx_list=[5, 9]):
        mask = None

        if torch.is_tensor(img):
            img = img.cpu().detach()
        if torch.is_tensor(logits):
            logits = logits.cpu().detach()

        for idx in idx_list:
            if mask is None:
                mask = self.get_part_mask(logits, idx)
            else:
                mask += self.get_part_mask(logits, idx)

        masked_img = cv2.bitwise_and(img, img, mask=mask)

        return masked_img

    def get_clothing_color(self, image, nr_of_clusters=3, image_to_rgb=False, return_text=True):
        if image_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_reshape = image.reshape((-1, 3)).astype(np.double)

        cluster_centers, _ = kmeans(image_reshape, nr_of_clusters)
        cluster_centers = np.round(cluster_centers, decimals=0).astype(int)
        # cluster_centers = cluster_centers.reshape(1, nr_of_clusters, 3) / 255
        # cluster_centers = np.round(cluster_centers * 255, decimals=0).astype(int)

        if return_text:
            colors = []
            for color in cluster_centers:
                # colors.append(convert_rgb_to_names(color, webcolors.CSS3_HEX_TO_NAMES))
                if color == 'black':
                    continue
                if color in colors:
                    continue

                colors.append(convert_rgb_to_names(color))
            return cluster_centers, colors
        else:
            return cluster_centers

    def get_orb_descriptors(self, img):
        # Check if image is grayscale
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.orb.detectAndCompute(img, None)

        return descriptors
