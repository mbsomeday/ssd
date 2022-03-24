def get_classes(cls_path):
    classes = open(cls_path).read().strip().split()
    return classes, len(classes)
    

























