from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from os import listdir
from datetime import datetime


class Annotator:
    def __init__(self, directory_root):
        self.object_classes = {0: 'Person'}
        # Create config
        cfg = get_cfg()
        cfg.merge_from_file("faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "model_final_f6e8b1.pkl"
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.root_directory = directory_root
        # Reading dataset
        self.dataset = self.read_images(directory_root)
        self.log_file = open('annot_test.csv', 'a+')

    def pred(self):
        timer = []
        timer1 = datetime.now()
        for image_name in self.dataset:
            timer2 = datetime.now()
            image = self.root_directory + '/' + image_name
            # print('processing ', image)
            # get image
            im = cv2.imread(image)
            # Make prediction
            # outputs = self.predictor(im)
            detections = self.predictor(im)
            # self.save_annotations(outputs, image_name)
            # log_file = open('annot_test.csv', 'a+')
            a = detections['instances'].pred_classes.cpu
            for (class_id, box) in zip(detections['instances'].pred_classes, detections['instances'].pred_boxes):
                if int(class_id) in self.object_classes.keys():
                    box = box.cpu()
                    left, top, right, bottom = box.numpy()
                    self.log_file.write(image_name + ',' + self.object_classes[int(class_id)] + ',' + str(left) + ',' + str(
                        top) + ',' + str(right) + ',' + str(bottom) + '\n')
            timer.append((datetime.now() - timer2).microseconds)
        self.log_file.close()
        print('overall time ', (datetime.now() - timer1))
        print("avg time per frame", sum(timer) /len(timer))

    def save_annotations(self, detections, image_name):
        log_file = open('annot_test.csv', 'a+')
        a = detections['instances'].pred_classes.cpu
        for (class_id, box) in zip(detections['instances'].pred_classes, detections['instances'].pred_boxes):
            if int(class_id) in self.object_classes.keys():
                box = box.cpu()
                left, top, right, bottom = box.numpy()
                log_file.write(image_name + ',' + self.object_classes[int(class_id)] + ',' + str(left) + ',' + str(
                    top) + ',' + str(right) + ',' + str(bottom) + '\n')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video streaming over zmq')
    parser.add_argument('--dir', type=str, default="/home/webwerks/PycharmProjects/annotator/coco_data")
    args = parser.parse_args()

    dataset_directory = Client(args.dir)
    annotator = Annotator()
    annotator.pred()
