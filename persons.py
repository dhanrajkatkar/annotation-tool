import cv2


class SsdModel:
    def __init__(self):
        # pretrained model (SSD mobilenet)
        model_file = "frozen_inference_graph.pb"
        pbtext_file = "ssd_mobilenet.pbtxt"
        # model loading
        self.model = cv2.dnn.readNetFromTensorflow(model_file, pbtext_file)
        # specific detection classes
        self.class_names = {1: 'person'}
        # Threashold value for detection confidence
        self.person_threashold = 0.5

    # method to predict gender
    def make_predict(self, image):
        # image = cv2.imread(image_directory)
        image_height, image_width, _ = image.shape
        image_copy = image.copy()
        self.model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
        # detection of all persons from image
        persons = self.model.forward()
        person_list = []
        for person in persons[0, 0, :, :]:
            confidence = person[2]
            if confidence > self.person_threashold:
                class_id = person[1]
                class_name = self.id_class_name(class_id, self.class_names)
                if class_name is None:
                    continue
                # print(str(str(class_id) + " " + str(person[2]) + " " + class_name))
                # conversion of person values to pixel coordinates
                left = int(person[3] * image_width)
                top = int(person[4] * image_height)
                right = int(person[5] * image_width)
                bottom = int(person[6] * image_height)
                person_list.append([left, top, right, bottom])
        return person_list

    # returns class index os class name
    @staticmethod
    def id_class_name(class_id, classes):
        for key, value in classes.items():
            if class_id == key:
                return value
