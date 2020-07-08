import numpy as np
from ground_truth_model import Detector
import cv2
from persons import SsdModel

class F1Score:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    @staticmethod
    def get_max_iou(pred_boxes, gt_box):
        # 1. calculate the inters coordinate
        if pred_boxes.shape[0] > 0:
            ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
            ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
            iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
            iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)

            # 2.calculate the area of inters
            inters = iw * ih
            # 3.calculate the area of union
            uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
                   (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
                   inters)
            # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
            iou = inters / uni
            iou_max = np.max(iou)
            nmax = np.argmax(iou)
            return iou, iou_max, nmax

    def calculate_score(self):
        precision = self.TP / (self.TP + self.FP)
        recall = self.TP / (self.TP + self.FN)
        f1_score = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1_score

    def confusion_matrix(self, gt_boxes, pred_boxes):
        if len(pred_boxes) < len(gt_boxes):
            print(len(pred_boxes), len(gt_boxes))
            self.FN += len(gt_boxes) - len(pred_boxes)
        for box in pred_boxes:
            iou, iou_max, nmax = f1.get_max_iou(gt_boxes, box)
            # print(iou_max)
            if iou_max > .5:
                self.TP += 1
            else:
                self.FP += 1
        print('TP = ', self.TP, 'FP =', self.FP, 'FN =', self.FN)


if __name__ == '__main__':
    f1 = F1Score()
    dir_root = '/home/webwerks/PycharmProjects/annotation-tool/coco_data'
    gt = Detector(0.5)
    # pred = Detector(0.7)
    pred = SsdModel()

    images = gt.read_images(dir_root)
    for image_name in images:
        image = dir_root + '/' + image_name
        # get image
        im = cv2.imread(image)
        gt_boxes = gt.pred(im)
        pred_boxes = pred.make_predict(im)
        f1.confusion_matrix(gt_boxes, pred_boxes)
    precision, recall, f1_score = f1.calculate_score()
    print(precision, recall, f1_score)
