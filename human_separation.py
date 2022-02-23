import cv2
import torch
import logging
import numpy as np

from tqdm import tqdm
from utils import get_dict, get_gt, xywh2xyxy, load_model, get_pred


class HumanSeparator():
    def __init__(self, video_path, gt_dict_path, load_segmentation=False, device=None):
        self.logger = logging.getLogger(__name__)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.segmentation = load_segmentation
        if load_segmentation:
            if device:
                self.device = device
            else:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                self.model = load_model(self.device)
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error when loading model: {e}")

        self.cap = cv2.VideoCapture(video_path)
        self.gt_dict = get_dict(gt_dict_path, separator=' ')

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames_nr = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.scaler = np.array([self.width, self.height,
                                self.width, self.height])

    def get_meta_info(self):
        meta = {'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'total_frame_nr': self.total_frames_nr}

        return meta

    def get_gt_dict(self):
        return self.gt_dict

    def get_cap(self):
        return self.cap

    def get_frame(self, frame_idx, frame_rgb=True):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = self.cap.read()
        if frame_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret is False:
            self.logger.info('cv2 VideoCapture finished extracting single frame')

        return frame

    def generate_crops(self, output_path, start_frame=None, end_frame=None, use_segmentation=False):
        if start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if end_frame is None:
            end_frame = self.total_frames_nr

        for curr_frame in tqdm(range(start_frame, end_frame)):
            idx = 0

            ret, frame = self.cap.read()
            if ret is False:
                self.logger.info('cv2 VideoCapture finished reading')

            # TODO:
            # Provide option for loading object detection model instead of dictionary
            try:
                dets_scores_classes = get_gt(curr_frame, self.gt_dict)
                self.logger.info("Detection dictionary loaded successfully")
            except Exception as e:
                self.logger.error(f"Error when loading detection dictionary: {e}")

            if (len(dets_scores_classes) == 3) and (None not in dets_scores_classes):
                detections = dets_scores_classes[0]
                out_scores = dets_scores_classes[1]
                classes = dets_scores_classes[2]

            if use_segmentation:
                labels = get_pred(frame, self.model)
                mask = labels == 15
                mask = np.array(mask, dtype=np.uint8) * 255
                frame = cv2.bitwise_or(frame, frame, mask=mask)

            for det, _, obj_cls in zip(detections, out_scores, classes):
                if int(obj_cls) == 0:
                    det = det * self.scaler
                    bbox = xywh2xyxy(det)
                    xmin, ymin, xmax, ymax = bbox.astype(int)
                    curr_person = frame[ymin:ymax, xmin:xmax]

                    cv2.imwrite(f'{output_path}/{curr_frame}_{idx}.png', curr_person)
                idx += 1

    def get_humans_from_frame(self, frame_idx, use_segmentation=False):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        humans = []

        ret, frame = self.cap.read()
        if ret is False:
            self.logger.info('cv2 VideoCapture finished extracting single frame')

        try:
            dets_scores_classes = get_gt(frame_idx, self.gt_dict)
            self.logger.info("Detection dictionary loaded successfully")
        except Exception as e:
            self.logger.error(f"Error when loading detection dictionary: {e}")

        if (len(dets_scores_classes) == 3) and (None not in dets_scores_classes):
            detections = dets_scores_classes[0]
            out_scores = dets_scores_classes[1]
            classes = dets_scores_classes[2]

            if use_segmentation:
                labels = get_pred(frame, self.model)
                mask = labels == 15
                mask = np.array(mask, dtype=np.uint8) * 255
                frame = cv2.bitwise_or(frame, frame, mask=mask)

            for det, _, obj_cls in zip(detections, out_scores, classes):
                if int(obj_cls) == 0:
                    det = det * self.scaler
                    bbox = xywh2xyxy(det)
                    xmin, ymin, xmax, ymax = bbox.astype(int)
                    curr_person = frame[ymin:ymax, xmin:xmax]
                    humans.append(curr_person)
        else:
            self.logger.warning(f"No detections available for frame {frame_idx}")

        return humans

    def get_frame_and_bboxs(self, frame_idx, use_segmentation=False, frame_rgb=True, return_mask=False):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = self.cap.read()
        if frame_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret is False:
            self.logger.info('cv2 VideoCapture finished extracting single frame')

        try:
            dets_scores_classes = get_gt(frame_idx, self.gt_dict)
            self.logger.info("Detection dictionary loaded successfully")
        except Exception as e:
            self.logger.error(f"Error when loading detection dictionary: {e}")

        if (len(dets_scores_classes) == 3) and (None not in dets_scores_classes):
            detections = dets_scores_classes[0]
            out_scores = dets_scores_classes[1]
            classes = dets_scores_classes[2]

            if return_mask or use_segmentation:
                labels = get_pred(frame, self.model, self.device)
                mask = labels == 15
                mask = np.array(mask, dtype=np.uint8) * 255

            if use_segmentation:
                frame = cv2.bitwise_or(frame, frame, mask=mask)

            if return_mask:
                return frame, detections, out_scores, classes, mask
            else:
                return frame, detections, out_scores, classes
        else:
            if return_mask:
                return [], [], [], [], []
            else:
                return [], [], [], []
