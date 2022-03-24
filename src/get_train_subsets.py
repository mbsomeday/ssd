import os
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET

from utils import utils


dataset_base_path = r'../../dataset\VOCdevkit\VOC2007'
annotationDir = 'Annotations'
imageDir = 'JPEGImages'
setDir = 'ssd_set'
cls_path = 'voc_classes.txt'

def makeDatasetDir():
    '''
    make dir to restore files: train.txt, test.txt, val.txt
    '''
    path = os.path.join(dataset_base_path, setDir)
    if not os.path.exists(path):
        os.mkdir(path)
        print('Generate set dir successfully!')
        
def get_split_set():
    '''
    split the original data
    '''
    tv_ratio = 0.9
    tr_ratio = 0.9
    
    image_path = os.path.join(dataset_base_path, imageDir)
    image_ids = list(os.listdir(image_path))
    total_images = len(image_ids)
    l = list(range(total_images))
    
    tv = int(tv_ratio * total_images)
    tr = int(tr_ratio * tv)
    
    trainVal = random.sample(l, tv)
    train = random.sample(trainVal, tr)
    
    set_dir = os.path.join(dataset_base_path, setDir)
    ftrainVal = open(os.path.join(set_dir, 'trainVal.txt'), 'w')
    ftrain = open(os.path.join(set_dir, 'train.txt'), 'w')
    ftest = open(os.path.join(set_dir, 'test.txt'), 'w')
    fval = open(os.path.join(set_dir, 'val.txt'), 'w')
    
    for id in tqdm(l):
        name = image_ids[id][: -4] + '\n'
        if id in trainVal:
            ftrainVal.write(name)
            if id in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    ftrainVal.close()
    ftrain.close()
    ftest.close()
    fval.close()

def get_anno_set():
    '''
    get .txt file for training
    '''
    annotation = os.path.join(dataset_base_path, annotationDir)
    
    ftrain = open(os.path.join(dataset_base_path, setDir, 'train.txt')).read().strip().split()
    ftest = open(os.path.join(dataset_base_path, setDir, 'test.txt')).read().strip().split()
    
    fannotrain = open('train.txt', 'w', encoding='utf-8')
    fannpval = open('val.txt', 'w', encoding='utf-8')
    
    for item in tqdm(ftrain):
        image_path = os.path.join(dataset_base_path, imageDir, item) + '.jpg'
        fannotrain.write(image_path)
        get_annotation(item, fannotrain)
        fannotrain.write('\n')
        
    for item in tqdm(ftest):
        image_path = os.path.join(dataset_base_path, imageDir, item) + '.jpg'
        fannpval.write(image_path)
        get_annotation(item, fannpval)
        fannpval.write('\n')

    fannotrain.close()
    fannpval.close()
    
def get_annotation(item, targetFile):
    '''
    write annotations to .txt file
    xmin, ymin, xmax, ymax, cls
    '''
    anno_path = os.path.join(dataset_base_path, annotationDir, item) + '.xml'
    anno = open(anno_path, encoding='utf-8')
    tree = ET.parse(anno)
    root = tree.getroot()
    
    classes, _ = utils.get_classes(cls_path)
    for obj in root.iter('object'):
        difficult = root.find('difficult').text if root.find('difficult') else 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')

        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        
        targetFile.write(' ' + ",".join(str(a) for a in b) + ',' + str(cls_id))
        
        
def preprocess_dataset(mode):
    '''
    :param mode: 0: both
                 1: only generate .txt in VOCdir
                 2: generate .txt(with annotation) for training
    :return:
    '''
    makeDatasetDir()
    if mode == 2:
        get_anno_set()
    elif mode == 1:
        get_split_set()
    else:
        get_split_set()
        get_anno_set()
        
    

if __name__ == '__main__':
    preprocess_dataset(2)





























