import cv2
import time
import numpy as np
import torch
import torchvision
import webcolors
from scipy.spatial import KDTree


def load_model(device):
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=True)
    model.to(device).eval()

    return model


def get_pred(img, model, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
    preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(
                                                mean=imagenet_stats[0],
                                                std=imagenet_stats[1])])
    input_tensor = preprocess(img).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Make the predictions for labels across the image
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
        output = output.argmax(0)

    # Return the predictions
    return output.cpu().numpy()


def get_dict(filename, separator: str = ','):
    with open(filename) as f:
        d = f.readlines()

    d = list(map(lambda x: x.strip(), d))

    last_frame = int(d[-1].split(separator)[0])

    gt_dict = {x: [] for x in range(last_frame + 1)}

    for i in range(len(d)):
        a = list(d[i].split(separator))
        a = list(map(float, a))

        coords = a[2:6]
        confidence = a[6]
        obj_class = a[1]

        gt_dict[a[0]].append(
            {'coords': coords, 'conf': confidence, 'class': obj_class})

    return gt_dict


def get_gt(frame_id, gt_dict):

    if frame_id not in gt_dict.keys() or gt_dict[frame_id] == []:
        return None, None, None

    frame_info = gt_dict[frame_id]

    detections = []
    ids = []
    out_scores = []
    classes = []

    for i in range(len(frame_info)):

        coords = frame_info[i]['coords']

        x1, y1, w, h = coords
        x2 = x1 + w
        y2 = y1 + h

        detections.append([x1, y1, w, h])
        out_scores.append(frame_info[i]['conf'])
        classes.append(frame_info[i]['class'])

    return detections, out_scores, classes


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y


def time_to_frame(mins, secs, fps=24.0):
    tmp_secs = mins * 60 + secs
    return int(tmp_secs * fps)


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_logits(logits, center, scale, width, height, input_size):
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i],
            trans,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits, axis=2)

    return target_logits


def convert_rgb_to_names(rgb_tuple, custom_palette=None):
    if custom_palette is None:
        custom_palette = {
            '#cc1f14': 'red',
            '#f78c84': 'light red',
            '#e08202': 'orange',
            '#ffb44d': 'light orange',
            '#f1d701': 'yellow',
            '#fff074': 'light yellow',
            '#3b9e1b': 'green',
            '#92d37d': 'light green',
            '#123ecf': 'blue',
            '#7293fe': 'light blue',
            '#b417ae': 'purple',
            '#e185df': 'light purple',
            '#4E3524': 'brown',
            '#000000': 'black',
            '#A59C94': 'gray',
            '#ffffff': 'white'}

    color_db = custom_palette
    names = []
    rgb_values = []
    for color_hex, color_name in color_db.items():
        names.append(color_name)
        rgb_values.append(webcolors.hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple, k=1, p=1)
    return names[index]


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # Example usage:
    # draw_border(img, (10,10), (100, 100), (127,255,255), 1, 10, 20)


def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                   fontScale=scale / 10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale / 12
    return 1


def frame_to_time(frame_nr, fps=24.0):
    seconds = frame_nr / fps
    ty_res = time.gmtime(seconds)
    res = time.strftime("%M:%S", ty_res)

    return res