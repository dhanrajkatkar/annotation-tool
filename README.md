# semi automatic image annotation
the script annotates all images specified in a directory.
it uses [detectron2](https://ai.facebook.com/tools/detectron2/) and faster RCNN.

### Requirements
* pytorch
* [Detectron2](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
* openCV
* faster rcnn r101 [model](https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl)

### Usage
```
    python detector.py -dir "path/to/data/folder"
```
