# from https://github.com/alicranck/coco2voc/tree/master
import os
import time

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from coco2voc_aux import annotations_to_seg


def coco2voc(annotations_file: str, target_folder: str, n: int = None, apply_border: bool = False,
             compress: bool = True):
    
    coco_instance = COCO(annotations_file)
    coco_imgs = coco_instance.imgs

    if n is None:
        n = len(coco_imgs)
        print(n)
    else:
        assert type(n) == int, "n must be an int"
        n = min(n, len(coco_imgs))

    instance_target_path = os.path.join(target_folder, 'instance_labels')
    class_target_path = os.path.join(target_folder, 'class_labels')
    id_target_path = os.path.join(target_folder, 'id_labels')

    os.makedirs(instance_target_path, exist_ok=True)
    os.makedirs(class_target_path, exist_ok=True)
    os.makedirs(id_target_path, exist_ok=True)

    image_id_list = open(os.path.join(target_folder, 'images_ids.txt'), 'a+')
    start = time.time()

    for i, img in enumerate(coco_imgs):

        annotation_ids = coco_instance.getAnnIds(img)
        annotations = coco_instance.loadAnns(annotation_ids)
        if not annotations:
            continue

        # class_seg, instance_seg, id_seg = annotations_to_seg(annotations, coco_instance, apply_border)
        class_seg= annotations_to_seg(annotations, coco_instance, apply_border)

        Image.fromarray(class_seg).convert("L").save(class_target_path + '/' + str(img).zfill(12) + '.png')
        # Image.fromarray(instance_seg).convert("L").save(instance_target_path + '/' + str(img) + '.png')
        
        # if compress:
            # np.savez_compressed(os.path.join(id_target_path, str(img)), id_seg)
        # else:
        #     np.save(os.path.join(id_target_path, str(img)+'.npy'), id_seg)

        image_id_list.write(str(img).zfill(12)+'\n')

        if i % 100 == 0 and i > 0:
            print(str(i) + " annotations processed" +
                  " in " + str(int(time.time()-start)) + " seconds")
        if i >= n:
            break

    image_id_list.close()
    return

if __name__ == '__main__':
    annotations_file = r'./annotations/instances_train2017.json'
    labels_target_folder = r'label_train'

    coco2voc(annotations_file, labels_target_folder, apply_border=False, compress=True)