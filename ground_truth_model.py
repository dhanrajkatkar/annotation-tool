from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
from os import listdir
import numpy as np


class Detector:
    def __init__(self, thresh):
        self.object_classes = {0: 'Person'}
        # Create config
        cfg = get_cfg()
        cfg.merge_from_file("faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh  # set threshold for this model
        cfg.MODEL.WEIGHTS = "model_final_f6e8b1.pkl"
        # Create predictor
        self.predictor = DefaultPredictor(cfg)

    def pred(self, im):
        boxes = []
        # Make prediction
        detections = self.predictor(im)
        a = detections['instances'].pred_classes.cpu
        for (class_id, box) in zip(detections['instances'].pred_classes, detections['instances'].pred_boxes):
            if int(class_id) in self.object_classes.keys():
                box = box.cpu()
                left, top, right, bottom = box.numpy()
                boxes.append((float(left), float(top), float(right), float(bottom)))
        return np.array(boxes)

    def save_annotations(self, detections, image_name):
        log_file = open('annot_test.csv', 'a+')
        a = detections['instances'].pred_classes.cpu
        for (class_id, box) in zip(detections['instances'].pred_classes, detections['instances'].pred_boxes):
            if int(class_id) in self.object_classes.keys():
                box = box.cpu()
                left, top, right, bottom = box.numpy()
                log_file.write(image_name + ',' + self.object_classes[int(class_id)] + ',' + str(box) + '\n')
        log_file.close()

    @staticmethod
    def read_images(directory_root):
        image_list = []
        print("[INFO] Image loading started")
        try:
            dataset_images = listdir(directory_root)
            for image in dataset_images:
                image_directory = f"{directory_root}/{image}"
                if image_directory.endswith(".jpg") is True or image_directory.endswith(".JPG") is True:
                    image_list.append(image)
            print("[INFO] Image loading completed")
            return image_list
        except Exception as e:
            print(f"Error : {e}")


