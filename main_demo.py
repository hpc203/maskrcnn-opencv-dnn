#!/usr/bin/python3
#!--*-- coding: utf-8 --*--
import cv2
import time
import numpy as np
import os
import random

class general_maskrcnn_dnn(object):
    def __init__(self, modelpath):
        self.conf_threshold = 0.5  # Confidence threshold
        self.mask_threshold = 0.3  # Mask threshold
        self.colors = [[  0., 255.,   0.],
                       [  0.,   0., 255.],
                       [255.,   0.,   0.],
                       [  0., 255., 255.],
                       [255., 255.,   0.],
                       [255.,   0., 255.],
                       [ 80.,  70., 180.],
                       [250.,  80., 190.],
                       [245., 145.,  50.],
                       [ 70., 150., 250.],
                       [ 50., 190., 190.], ]

        self.maskrcnn_model = self.get_maskrcnn_net(modelpath)
        self.classes = self.get_classes_name()


    def get_classes_name(self):
        # Load names of classes
        classesFile = "mscoco_labels.names"
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        return classes


    def get_maskrcnn_net(self, modelpath):
        pbtxt_file = os.path.join(modelpath, './graph.pbtxt')
        pb_file    = os.path.join(modelpath, './frozen_inference_graph.pb')

        maskrcnn_model = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
        maskrcnn_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        maskrcnn_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return maskrcnn_model


    def postprocess(self, boxes, masks, img_height, img_width):
        # 对于每个帧，提取每个检测到的对象的边界框和掩码
        # 掩模的输出大小为NxCxHxW
        # N  - 检测到的框数
        # C  - 课程数量（不包括背景）
        # HxW  - 分割形状
        numClasses = masks.shape[1]
        numDetections = boxes.shape[2]

        results = []
        for i in range(numDetections):
            box = boxes[0, 0, i]
            mask = masks[i]
            score = box[2]
            if score > self.conf_threshold:
                left = int(img_width * box[3])
                top = int(img_height * box[4])
                right = int(img_width * box[5])
                bottom = int(img_height * box[6])

                left = max(0, min(left, img_width - 1))
                top = max(0, min(top, img_height - 1))
                right = max(0, min(right, img_width - 1))
                bottom = max(0, min(bottom, img_height - 1))

                result = {}
                result["score"]  = score
                result["classid"] = int(box[1])
                result["box"]   = (left, top, right, bottom)
                result["mask"]   = mask[int(box[1])]

                results.append(result)

        return results


    def predict(self, imgfile):
        img_cv2 = cv2.imread(imgfile)
        img_height, img_width, _ = img_cv2.shape

        # 从框架创建4D blob。
        blob = cv2.dnn.blobFromImage(img_cv2, swapRB=True, crop=True)

        # 设置网络的输入
        self.maskrcnn_model.setInput(blob)

        # 运行正向传递以从输出层获取输出
        boxes, masks = self.maskrcnn_model.forward(
            ['detection_out_final', 'detection_masks'])

        # 为每个检测到的对象提取边界框和蒙版
        results = self.postprocess(boxes, masks, img_height, img_width)

        return results


    def vis_res(self, img_file, results):
        img_cv2 = cv2.imread(img_file)

        for result in results:
            # box
            left, top, right, bottom = result["box"]
            cv2.rectangle(img_cv2,
                          (left, top),
                          (right, bottom),
                          (255, 178, 50), thickness=2)

            # class label
            classid = result["classid"]
            score   = result["score"]
            label = '%.2f' % score
            if self.classes:
                assert (classid < len(self.classes))
                label = '%s:%s' % (self.classes[classid], label)

            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, label_size[1])

            cv2.rectangle(
                img_cv2,
                (left, top - round(1.5 * label_size[1])),
                (left + round(1.5 * label_size[0]), top + baseline),
                (255, 255, 255), cv2.FILLED)
            cv2.putText(
                img_cv2,
                label,
                (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

            # mask
            class_mask = result["mask"]
            class_mask = cv2.resize(class_mask, (right - left + 1, bottom - top + 1))
            mask = (class_mask > self.mask_threshold)
            roi = img_cv2[top: bottom + 1, left: right + 1][mask]

            # color = self.color[classId%len(colors)]
            color_index = random.randint(0, len(self.colors) - 1)
            color = self.colors[color_index]

            img_cv2[top: bottom + 1, left: right + 1][mask] = (
                        [0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

            mask = mask.astype(np.uint8)
            contours, hierachy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(
                img_cv2[top: bottom + 1, left: right + 1],
                contours,
                -1,
                color,
                3,
                cv2.LINE_8,
                hierachy,
                100)

        t, _ = self.maskrcnn_model.getPerfProfile()
        label = 'Inference time: %.2f ms' % \
            (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(img_cv2, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.namedWindow('MaskRCNN detect', cv2.WINDOW_NORMAL)
        cv2.imshow('MaskRCNN detect', img_cv2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("[INFO]MaskRCNN detection.")

    img_file = "images/example_02.jpg"
    #
    start = time.time()
    modelpath =  "mask_rcnn_inception_v2_coco_2018_01_28"
    maskrcnn_model = general_maskrcnn_dnn(modelpath)
    print("[INFO]Model loads time: ", time.time() - start)

    start = time.time()
    res = maskrcnn_model.predict(img_file)
    print("[INFO]Model predicts time: ", time.time() - start)
    maskrcnn_model.vis_res(img_file, res)

    print("[INFO]Done.")
