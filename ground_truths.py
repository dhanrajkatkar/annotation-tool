from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
from os import listdir

class Detector:
    def __init__(self):
        self.object_classes = {0: 'Person'}
        # Create config
        cfg = get_cfg()
        cfg.merge_from_file("faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "model_final_f6e8b1.pkl"
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.log_file = open('ground_truth.csv', 'w')
        # self.log_file = open('predictions.csv', 'w')
        self.log_file.write('image name,object class,xmin,ymin,w,h\n')

    def pred(self):
        images = obj.read_images(dir_root)
        for image in images:
            image_file = dir_root + '/' + str(image)
            # get image
            im = cv2.imread(image_file)
            # Make prediction
            detections = self.predictor(im)
            # log_file = open('annot_test.csv', 'a+')
            a = detections['instances'].pred_classes.cpu
            boxes = []
            for (class_id, box) in zip(detections['instances'].pred_classes, detections['instances'].pred_boxes):
                if int(class_id) in self.object_classes.keys():
                    box = box.cpu()
                    left, top, right, bottom = box.numpy()
                    # boxes.append([str(left), str(top), str(right), str(bottom)])
                    self.log_file.write(image + ',' + self.object_classes[int(class_id)] + ',' + str(left) + ',' + str(top) + ',' + str(right) + ',' + str(bottom) + '\n')
        self.log_file.close()
        return boxes

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

    def save_gt(self, boxes, image_name):
        log_file = open('predictions.csv', 'a+')
        log_file.write(image_name + ',' + str(boxes) + '\n')
        log_file.close()

    @staticmethod
    def bb_intersection_over_union(box_a, box_b):
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        # compute the area of intersection rectangle
        intersection = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection / float(box_a_area + box_b_area - intersection)
        # return the intersection over union value
        return iou

if __name__ == '__main__':
    dir_root = '/home/webwerks/PycharmProjects/annotation-tool/coco_data'
    obj = Detector()
    boxes = obj.pred()
