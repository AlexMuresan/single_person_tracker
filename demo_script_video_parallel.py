import os
import cv2
import copy
import torch
import warnings
import numpy as np

from threading import Thread
from datetime import datetime

from tqdm import tqdm
from utils import get_gt
from utils import get_pred
from utils import load_model
from utils import xywh2xyxy

from utils import draw_border, get_optimal_font_scale

from human_separation import HumanSeparator
from feature_extraction import FeatureExtractor
from feature_comparison import Comparator

warnings.filterwarnings('ignore')


def process_video_part(thread_idx, start_frame, end_frame, model, ref_features):
    device = 'cuda:0'
    cap = cv2.VideoCapture('./data/video/AVSS_AB_Hard.mp4')
    curr_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

    refs, refs_torch, refs_hist, refs_nn_feats, refs_limbs_hist = ref_features

    while curr_frame != end_frame:
        ret, frame = cap.read()
        curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if curr_frame % 50 == 0:
            print(f'Thread {thread_idx}: Processing frame {curr_frame}')

        if (ret is True) or (frame is not None):
            frame_raw = copy.deepcopy(frame)
            dets_scores_classes = get_gt(curr_frame, gt_dict)

            if (len(dets_scores_classes) == 3) and (None not in dets_scores_classes):
                detections = dets_scores_classes[0]
                out_scores = dets_scores_classes[1]
                classes = dets_scores_classes[2]

                # Creating a mask based on our segmentation model
                # We apply person segmentation on the whole frame
                labels = get_pred(frame, model, device)
                mask = labels == 15
                mask = np.array(mask, dtype=np.uint8) * 255
                frame = cv2.bitwise_or(frame, frame, mask=mask)

                for det, _, obj_cls in zip(detections, out_scores, classes):
                    # For each frame we're only using the 'person' detections (0)
                    if int(obj_cls) == 0:
                        # Yolo detections need to be scaled back to the dimensions of the video
                        det = det * scaler
                        bbox = xywh2xyxy(det)
                        xmin, ymin, xmax, ymax = bbox.astype(int)
                        # We extract each human from the segmentated frame
                        human = frame[ymin:ymax, xmin:xmax]
                        # We also keep the detection without any segmentation
                        human_no_seg = frame_raw[ymin:ymax, xmin:xmax]

                        if np.mean(human) <= 21:
                            continue

                        human_original = copy.deepcopy(human_no_seg)
                        human = cv2.resize(human, dims, interpolation=cv2.INTER_CUBIC)

                        # Here we're extracting the same features we extracted for the refference images
                        # for each 'person' detection we have in the current frame
                        human_hist = feat_ext.color_histogram(human)
                        human_nn_feats = feat_ext.nnet_features(human, cpu_output=False)
                        human_logits = feat_ext.get_human_parsing_logits(human)
                        human_limbs = feat_ext.get_masked_limbs(human, human_logits, idx_list=[5, 7, 13])
                        human_limbs_hist = feat_ext.color_histogram(human_limbs)

                        scores = []

                        # Here we're comparing the refference features to the current detection's features
                        # We have two types of comparisons:
                        #   - cosine similarity
                        #   - histogram comparison
                        for ref, ref_torch, ref_hist, ref_nn_feats, ref_limbs_hist in zip(
                                refs, refs_torch, refs_hist, refs_nn_feats, refs_limbs_hist):

                            hist_score = comp.compare_histograms(human_hist, ref_hist)
                            cos_score = comp.cosine_similarity(torch.Tensor(human).to(device), ref_torch,
                                                            use_torch=True)
                            nn_feats_score = comp.cosine_similarity(human_nn_feats, ref_nn_feats,
                                                                    use_torch=True)
                            limbs_hist_score = comp.compare_histograms(human_limbs_hist, ref_limbs_hist)

                            # Sometimes the limbs separation doesn't work well, in which case we don't include that
                            # in the final score of the detection

                            if np.mean(human_limbs) < 15:
                                score = hist_score + cos_score.cpu() + nn_feats_score.cpu()
                            else:
                                score = hist_score + cos_score.cpu() + nn_feats_score.cpu() + limbs_hist_score
                            scores.append(score)

                        human_mask = mask[ymin:ymax, xmin:xmax]

                        zeros = np.zeros_like(human_mask)
                        ones = np.ones_like(human_mask)

                        # In case we have a high score (meaning the detection is close to the refference images)
                        # We draw the outline of the detected person with red.
                        # If the score is low the outline will be green
                        if np.mean(scores) >= 2.2:
                            color = np.stack([zeros, zeros, ones * 100], axis=2)
                            color_mask = cv2.bitwise_and(color, color, mask=human_mask)
                            masked_human = cv2.bitwise_or(human_original, color_mask)
                            frame_raw[ymin:ymax, xmin:xmax] = masked_human

                            optimal_font_scale = get_optimal_font_scale('Suspect', ((xmax - xmin) + 25))

                            xPos = xmin - 5
                            yPos = ymin - 10
                            while yPos < 10:
                                yPos += 10

                            draw_border(frame_raw, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2,
                                        r=5, d=5)
                            cv2.putText(frame_raw, 'Suspect', (xmin, yPos), cv2.FONT_HERSHEY_SIMPLEX,
                                        optimal_font_scale, (0, 0, 255), 1)
                        else:
                            color = np.stack([zeros, ones * 100, zeros], axis=2)
                            color_mask = cv2.bitwise_and(color, color, mask=human_mask)
                            masked_human = cv2.bitwise_or(human_original, color_mask)
                            frame_raw[ymin:ymax, xmin:xmax] = masked_human

                            draw_border(frame_raw, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2,
                                        r=5, d=5)

                cv2.imwrite(f'video_images/{int(curr_frame)}.jpg', frame_raw)

            else:
                print(f'Thread {thread_idx}: No detections found at frame {curr_frame}')
                continue

    print(f'Thread {thread_idx} done!')


def get_ref_features(device):
    # Scanning the reference image folder and extracting features from each image present
    # The extracted features are:
    #       - color histogram
    #       - neural net extracted features
    #       - color histogram extracted from torso, hands, legs and face

    refs = []
    refs_torch = []
    refs_hist = []
    refs_nn_feats = []
    refs_limbs_hist = []

    for i in tqdm(range(4)):
        ref = cv2.resize(cv2.imread(f'ref_images/{i}.png'), dims, interpolation=cv2.INTER_CUBIC)
        torch_ref = torch.Tensor(ref).to(device)

        refs.append(ref)
        refs_torch.append(torch_ref)

        refs_hist.append(feat_ext.color_histogram(ref))
        refs_nn_feats.append(feat_ext.nnet_features(ref, cpu_output=False))
        ref_logits = feat_ext.get_human_parsing_logits(ref, use_gpu=True)
        ref_limbs = feat_ext.get_masked_limbs(ref, ref_logits, idx_list=[5, 7, 13])
        refs_limbs_hist.append(feat_ext.color_histogram(ref_limbs))

    return (refs, refs_torch, refs_hist, refs_nn_feats, refs_limbs_hist)


def get_splits(start, end, num_threads):
    # Splitting video frames based on number of threads
    # Each thread is gettting is getting basically the same number of frames to process

    splits = [start]
    total_frames = end - start
    for i in range(1, num_threads):
        splits.append(start + i * int(total_frames/num_threads))

    splits.append(end)

    return splits

if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    device = 'cuda:0'

    # Instantiation of the human separator (we use this for segmentation and to parse detections from Yolo)
    sep = HumanSeparator('./data/video/AVSS_AB_Hard.mp4',
                        './data/detections/AVSS_AB_Hard.txt',
                        load_segmentation=True, device=device)

    scaler = sep.scaler
    cap = sep.get_cap()
    gt_dict = sep.get_gt_dict()
    meta = sep.get_meta_info()

    # Instantiation of the feature extractor
    feat_ext = FeatureExtractor(load_human_segmentation_model=True, load_human_parsing_model=True,
                                device=device)

    # Instantiation of the feature comparator
    comp = Comparator()

    model = load_model(device)

    # The dimensions at which we reshape each detection since this is what the neural network uses
    dims = (473, 473)

    # refs, refs_torch, refs_hist, refs_nn_feats, refs_limbs, refs_limbs_hist, refs_clothing_colors, ref_orb_descriptors = get_ref_features(device)
    ref_features = get_ref_features(device)

    # We're using 20 threads and starting at frame number 250 since the video is empty before that
    num_threads = 20
    start = 250
    finish = meta['total_frame_nr']

    splits = get_splits(start, finish, num_threads)

    threads = []

    # Starting each thread and giving it it's respective frame-range to process
    for i in range(num_threads):
        worker = Thread(target=process_video_part, args=(i, splits[i], splits[i+1], model, ref_features))
        worker.setDaemon(True)
        print(f'Starting thread {i}')
        worker.start()
        threads.append(worker)

    # Joining threads so we can keep monitoring them and so that the script won't finish
    # before the threads finish their work
    for t in threads:
        t.join()

    end_time = datetime.now()
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Run time: {end_time - start_time}")

    print("Running ffmpeg command to stitch images")
    ffmpeg_comand = f"ffmpeg -framerate 25 -pattern_type glob -i './video_images/*.jpg' -c:v mpeg4 ./output/AVSS_AB_Hard_Out.mp4 -b 5000k"
    os.system(ffmpeg_comand)