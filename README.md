# YOLO-DLv3plus-articulators-segmentation

code for mouth, lips, teeth, and tongue detection and segmentation using YOLOv6 & DeepLabv3+ (semi-supervised approach) 

**Sage A, Badura P. Detection and segmentation of mouth region in stereo stream using YOLOv6 and DeepLab v3+ models for computer-aided speech diagnosis in children. Applied Sciences-Basel. 2024;14:1–20. doi:10.3390/app14167146**


1) yolov6: code for trainig and prediction mouth, teeth, and tongue in camera images using YOLOv6 (pytorch) 
2) rough_segmentation: to prepare rough delineations of mouth, teeth, and tongue using DRLSE method. The results are the training dataset for inital trainig of DeepLabv3+ (as we had small number of expert's ground truth masks).
3) dlv3plus: code for trainig (and predicting) mouth, teeth, and tongue in camera images based on ROIs detected by yolov6 using a DeepLabv3+ architecture (pytorch)
