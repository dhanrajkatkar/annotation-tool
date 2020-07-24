from pycocotools.coco import COCO
import requests
import os
from datetime import datetime, timedelta

# configurations
required_categories = {8: 'truck'}
# annFile = "../instances_train2017.json"
annFile = "../instances_val2017.json"
folder_img_count = 500

# initialize COCO api for instance annotations
coco = COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catNms = [category for category in required_categories.values()]
catIds = coco.getCatIds(catNms=catNms)
imgIds = coco.getImgIds(catIds=catIds)
total_images = len(imgIds)
print(total_images)
images = coco.loadImgs(imgIds)

starting_index = 0
counter = starting_index
timer = datetime.now()
for img in images:
    folder_name = 'coco_data/truck/coco_v' + str(counter // folder_img_count)
    filename = folder_name + '/' + img['file_name']

    if os.path.isfile(filename):
        counter += 1
        continue

    print(counter, ": ", img['file_name'])

    if counter % 50 == 0:
        tt = (datetime.now() - timer).seconds
        remaining_img = (starting_index + total_images - counter)
        print('time required for 50 images ', tt, 'remaining time for ', remaining_img, ' images ',
              remaining_img * timedelta(seconds=tt / 50))
        timer = datetime.now()

    img_data = requests.get(img['coco_url']).content

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        with open(folder_name + "/annotations_v" + str(counter // folder_img_count) + '.csv', "a+") as myfile:
            myfile.write('file_name,classes,xmin,ymin,xmax,ymax\n')

    with open(filename, 'wb') as handler:
        handler.write(img_data)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
        anns = coco.loadAnns(annIds)
        with open(folder_name + "/annotations_v" + str(counter // folder_img_count) + '.csv', "a+") as myfile:
            for i in range(len(anns)):
                xmin = anns[i]["bbox"][0]
                ymin = anns[i]["bbox"][1]
                xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
                ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

                mystring = img['file_name'] + ',' + required_categories[anns[i]['category_id']] + ',' + str(
                    xmin) + ',' + str(ymin) + ',' + str(xmax) + "," + str(ymax)
                myfile.write(mystring + '\n')
    counter += 1
