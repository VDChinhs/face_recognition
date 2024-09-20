import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F

from modules.globals import anti_spoof_models, caffe_model, deploy_file

from modules.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from modules.model_lib.MultiFTNet import MultiFTNet
from modules.data_io import transform as trans
from modules.generate_patches import CropImage
from modules.utility import parse_model_name, get_kernel

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE,
    'MultiFTNet': MultiFTNet
}

class Detection:
    def __init__(self):
        caffemodel = caffe_model
        deploy = deploy_file
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        # max_conf_index = np.argmax(out[:, 2])
        listbbox = []
        for face in out:
            if face[1] == 1:
                left, top, right, bottom = face[3]*width, face[4]*height, \
                                   face[5]*width, face[6]*height
                bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
                listbbox.append(bbox)
        # left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
        #                            out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        # bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return listbbox

class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result, dim=-1).cpu().numpy()
        return result

def verify_face_real(frame, face):
    model_dir = anti_spoof_models

    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(frame)
    bbox = bbox_nearest(face.bbox, image_bbox)
    if bbox == None:
        return 2,1
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    return label, value

def calculate_center(bbox, type = 'xy'):
    if type == 'xy':
        x1, y1, x2, y2 = bbox
        center_x = (x2 - x1) / 2
        center_y = (y2 - y1) / 2
        return np.array([center_x, center_y])
    if type == 'wh':
        x, y, w, h = bbox
        center_x = w / 2
        center_y = h / 2
        return np.array([center_x, center_y])

def bbox_nearest(bbox, bbox_list):
    bbox_center = calculate_center(bbox, type='xy')
    centers = [calculate_center(b, type='wh') for b in bbox_list]

    distances = [np.linalg.norm(bbox_center - center) for center in centers]
    min_index = np.argmin(distances)
    if distances[min_index] < 50:
        return bbox_list[min_index]
    return None

def draw_face_spoof(frame, save_img = False):
    iframe = frame.copy()

    model_dir = anti_spoof_models
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(frame)
    for bbox in image_bbox:
        prediction = np.zeros((1, 3))
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        if label == 1:
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        cv2.rectangle(
            iframe,
            (param["bbox"][0], param["bbox"][1]),
            (param["bbox"][0] + param["bbox"][2], param["bbox"][1] + param["bbox"][3]),
            color, 2)
        cv2.putText(
            iframe,
            result_text,
            (param["bbox"][0], param["bbox"][1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/1024, color)
    if save_img:
        cv2.imwrite('inputs/result.jpg',iframe)
    return iframe