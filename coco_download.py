from pycocotools.coco import COCO
import numpy as np
import requests

annFile = "instances_val2017.json"

# initialize COCO api for instance annotations
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
print(len(imgIds))
images = coco.loadImgs(imgIds)

i = 0
for img in images:
    print("im: ", img['file_name'])
    img_data = requests.get(img['coco_url']).content
    if i > 200:
        break
    else:
        i += 1
    with open(img['file_name'], 'wb') as handler:
        handler.write(img_data)
