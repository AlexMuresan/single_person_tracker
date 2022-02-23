import cv2
import torch
import logging
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class Comparator():
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _get_part_mask(self, logits, idx):
        mask = np.zeros((logits.shape[0], logits.shape[1]), dtype=np.uint8)
        mask_idx = np.where(logits == idx)

        for x, y in zip(mask_idx[0], mask_idx[1]):
            mask[x][y] = 1

        return mask

    def compare_histograms(self, rgb_histogram_1, rgb_histogram_2, comp_method=cv2.HISTCMP_CORREL):
        scores = []

        for hist_1_channel, hist_2_channel in zip(rgb_histogram_1, rgb_histogram_2):
            scores.append(cv2.compareHist(hist_1_channel, hist_2_channel, comp_method))

        return np.mean(scores)

    def cosine_similarity(self, features_1, features_2, use_torch=False, cpu_output=False):
        if use_torch:
            self.logger.info('Using torch implementation of cosine similarity')
            score = self.cosine_sim(features_1, features_2).mean()
        else:
            self.logger.info('Using sklearn implementation of cosine similarity')
            score = cosine_similarity(features_1.reshape(1, -1), features_2.reshape(1, -1))[0][0]

        if cpu_output and use_torch:
            score = score.cpu().detach()

        return score

    def color_clothing_comparison(self, ref_colors, human_colors):
        matching_colors = 0

        for color in human_colors:
            if color in ref_colors:
                matching_colors += 1

        return matching_colors / len(ref_colors)

    def orb_matcher(self, img_descriptor_1, img_descriptor_2, return_mean=False):
        matches = self.bf.match(img_descriptor_1, img_descriptor_2)

        if return_mean:
            return np.mean([x.distance for x in matches])
        else:
            return matches
