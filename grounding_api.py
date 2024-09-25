def script_method(fn, _rcb=None):
    return fn

def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj

import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script

import numpy as np
import torch
from torchvision.ops import box_convert
import datetime
import xml.etree.ElementTree as ET
import cv2
import json
import os
from groundingdino.util.inference import load_model, load_image, predict
# from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
from argparse import ArgumentParser
from glob import glob
import shutil
import time
import logging

def create_xml(objects, filename_text, folder_text, width_text, height_text, save_path, save_xml=True):
    annotation = ET.Element("annotation")
    # 添加子節點
    folder = ET.SubElement(annotation, "folder")
    folder.text = folder_text
    filename = ET.SubElement(annotation, "filename")
    filename.text = filename_text
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = width_text
    height = ET.SubElement(size, "height")
    height.text = height_text
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    for obj in objects:
        object_node = ET.SubElement(annotation, "object")
        name = ET.SubElement(object_node, "name")
        name.text = obj["name"]
        bndbox = ET.SubElement(object_node, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = obj["xmin"]
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = obj["xmax"]
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = obj["ymin"]
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = obj["ymax"]
        truncated = ET.SubElement(object_node, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(object_node, "difficult")
        difficult.text = "0"
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    # 創建 ElementTree 物件
    tree = ET.ElementTree(annotation)
    # 將樹寫入文件
    if save_xml:
        tree.write(os.path.join(save_path, filename_text.split('.')[0]+'.xml'))
        print(os.path.join(save_path, filename_text.split('.')[0]+'.xml'))

    return tree

def create_non_xml(objects, filename_text, folder_text, width_text, height_text, save_path, save_xml=True):
    root = ET.Element("root")
    tree = ET.ElementTree(root)
    tree.write(os.path.join(save_path, filename_text.split('.')[0]+'.xml'))

def bbox_convert(boxes,image_source):
    '''
    Filter background with opencv match template, 
    '''
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    # detections = sv.Detections(xyxy=xyxy)
    return xyxy


def non_maximum_suppression_fast(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    pick = []
    x1 = boxes[:,0].astype("float")
    y1 = boxes[:,1].astype("float")
    x2 = boxes[:,2].astype("float")
    y2 = boxes[:,3].astype("float")
    bound_area = (x2-x1+1) * (y2-y1+1)
    sort_index = np.argsort(y2)
    while sort_index.shape[0] > 0:
        last = sort_index.shape[0]-1
        i = sort_index[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[sort_index[:last]])
        yy1 = np.maximum(y1[i], y1[sort_index[:last]])
        xx2 = np.minimum(x2[i], x2[sort_index[:last]])
        yy2 = np.minimum(y2[i], y2[sort_index[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w*h) / bound_area[sort_index[:last]]
        sort_index = np.delete(sort_index, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))) 
    return boxes[pick]


def write_json(image, filename_element, all_contours, save_path,label_name):
    d = {
        "version": "4.2.9",
        "flags": {},
        "shapes": [],
        "imagePath":filename_element ,
        "imageData": None,
        "imageHeight": image.shape[0],
        "imageWidth": image.shape[1]
        }
    shape = []
    points = []
    for contours in all_contours:
        for contour in contours:
            la = {"label": label_name, "text": "", "points":None , "group_id": None, "shape_type": "polygon", "flags": {}}
            points = contour.reshape((-1, 2)).tolist()
            if len(points) > 2:
                if cv2.contourArea(contour)>8:
                    la["points"] = points
                    shape.append(la)
    d["shapes"] = shape
    ph_j = os.path.join(save_path , os.path.splitext(filename_element)[0]+'.json')
    print('File to:',ph_j)
    with open(ph_j , "w") as f:
        json.dump(d, f)

def write_non_json(filename_element, save_path):
    d ={}
    ph_j = os.path.join(save_path , os.path.splitext(filename_element)[0]+'.json')
    print('File to:',ph_j)
    with open(ph_j , "w") as f:
        json.dump(d, f)

def show_mask(masks, image, input_point, boxes ,txt, output_dir,random_color=False):
    '''Plot masks on the image'''
    for i, mask in enumerate(masks):
        if random_color:
            colors = np.random.random(3)*250
            color = np.array([int(k) for k in colors],dtype=np.uint8) 
        else:
            color = np.array([200, 0, 0],dtype=np.uint8)
        
        h, w = mask.shape[0], mask.shape[1]
        mask_image = mask.reshape(h, w, 1).astype(np.uint8) * color.reshape(1, 1, -1)
        mask =  np.uint8(masks[0])
        # ret, mask0  = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY_INV)
        # image = cv2.bitwise_and(image, image, mask = mask0)
        image = cv2.addWeighted(image, 0.7, mask_image, 0.4, 30)
        if boxes is not None:
            for box in boxes:
                box0 = (int(box[0]), int(box[1]))
                box1 = (int(box[2]), int(box[3]))
                image = cv2.rectangle(image, box0, box1,(100, 50, 0), 1)
        if input_point is not None:
            if input_point.any():
                point = (int(input_point[0]),int(input_point[1]))
                image = cv2.circle(image, point, 3, (100, 50, 0),-1)
        if txt:
            for i in range(len(boxes)):
                text = 'IOUs:{:.3f}'.format(txt[i])
                xx = int(boxes[i][0])-5
                yy =  int(boxes[i][1])-5
                cv2.putText(image, text, (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imwrite(output_dir, image)


def ann_pic(box_im, image_source,  labels):
    detections = sv.Detections(xyxy=np.array(box_im))
    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def show_messqge(mes):
    print(datetime.datetime.now(),mes)

def get_imgfile(Input):
    path0 = glob(Input+'\\*.jpg')
    path1 = glob(Input+'\\*.png')
    return path0+path1

def remove_file(Dir):
    '''Remove Dir/*'''
    filelist = glob(os.path.join(Dir, "*"))
    for f in filelist:
        os.remove(f)

def check_tmp(Dir,checkpoint):
    if os.path.isdir(os.path.join(Dir,checkpoint)):
        return os.path.join(Dir,checkpoint)
    else:
        os.makedirs(os.path.join(Dir,checkpoint))
        return os.path.join(Dir,checkpoint)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_folder",default='D:\\Uploads\\Groundingdino_input')
    parser.add_argument("--output_folder",default='D:\\Uploads')
    parser.add_argument("--device", default="cuda:0")

    # parser.add_argument("--SAM_checkpoints", default="weights/sam_pb4.pth")
    parser.add_argument("--GD_checkpoints", default="weights/checkpoint.pth")
    parser.add_argument("--GD_cinfig_py", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--TEXT_PROMPT", default='defect')
    parser.add_argument("--BOX_TRESHOLD", type=float, default=0.2)
    parser.add_argument("--TEXT_TRESHOLD",  type=float, default=0.2)
    args = parser.parse_args()
    print(args)
    TORCH_CUDA_ARCH_LIST="6.0"
    log_path = os.path.join('log')
    if not os.path.isdir(log_path ):
        os.makedirs(log_path)

    logging.basicConfig(level = logging.DEBUG,
                format = '[%(levelname)s] %(asctime)s %(message)s',
                datefmt ='%Y-%m-%d %H:%M:%S',
                filename = os.path.join(log_path,str(os.getpid())+'.log'),
                filemode = 'a')
    
    logger = logging.getLogger(__name__)
    show_messqge('Loading model...')
    device = args.device
    model = load_model( args.GD_cinfig_py, args.GD_checkpoints, device=device)
    # sam_checkpoint = args.SAM_checkpoints
    # model_type = "vit_l"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    # predictor = SamPredictor(sam)
    show_messqge('Loaded model !')
    print(torch.cuda.get_arch_list())
    # Groundingdino(GD) autolabel to .xml
    # step: detect(GD) -> nms -> match template (del background) -> xml
    TEXT_PROMPT = args.TEXT_PROMPT
    BOX_TRESHOLD = args.BOX_TRESHOLD
    TEXT_TRESHOLD = args.TEXT_TRESHOLD
    Input = args.input_folder
    endpoint = 'D:\\Autolabeling\\endpoint'
    tmp_path = os.path.join('tmp')
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)
    else:
        remove_file(tmp_path)
    if not os.path.isdir(endpoint):
        os.makedirs(endpoint)
    else:
        remove_file(endpoint)
    while True:
        time.sleep(0.1)
        show_messqge('Waiting...')
        paths = glob(Input+'\\*.txt')
        if len(paths) > 0:
            for txt_path0 in paths:
                try:
                    show_messqge('Get file {}'.format(txt_path0))
                    if txt_path0.split('_')[-1] == 'end.txt':
                        # 20240718_1048_LCD1_4056_8338_end.txt
                        txt_ = txt_path0.split('\\')[-1]
                        fab, md_id, id = txt_.split('_')[2], txt_.split('_')[3], txt_.split('_')[4]
                        end_dir = os.path.join(args.output_folder, fab, md_id, 'Labeling', id, 'End')
                        if not os.path.isdir(end_dir):
                            os.makedirs(end_dir)
                        try:
                            endpoint_path = shutil.move(os.path.join(txt_path0), end_dir)
                            print(endpoint_path)
                        except:
                            logger.error('End_file has already exist', txt_path0)
                            os.remove(txt_path0)
                    else:
                        txt_path = shutil.move(os.path.join(txt_path0), tmp_path) # 先將抓到的txt移到tmp
                        time.sleep(0.001)
                        with open(txt_path, 'r', encoding='UTF-8') as f:
                            lines = f.readlines()
                            for txt in lines:
                                # try:  d:\uploads\LCD1\4056\Labeling\8338\pic\3350-1T79CV968AM050011.JPG,HDC/R,LCD1,4056,8338
                                # D:\Uploads\LCD1\4056\Labeling\8338
                                if len(txt.split(',')) == 5:
                                    fab = txt.split(',')[2]
                                    md_id = txt.split(',')[3]
                                    id = txt.split(',')[4]
                                    # out_dir = 'C:\\Users\yangu\\Desktop\\123\\{}\\output'.format(id)
                                    out_dir = os.path.join(args.output_folder, fab, md_id, 'Labeling', id, 'Label')
                                    label_name = txt.split(',')[1]
                                    n_file = False
                                    if not os.path.isdir(out_dir):
                                        os.makedirs(out_dir)
                                else:
                                    # out_dir = 'C:\\Users\yangu\\Desktop\\123\\output'
                                    out_dir = os.path.join(args.output_folder, 'Autolabeling_tmp', 'Groundingdino_Label') #'D:\\Uploads\\autolabeling\\Label'
                                    label_name = 'defect'
                                    n_file = True
                                    if not os.path.isdir(out_dir):
                                        os.makedirs(out_dir)
                                
                                img_path = os.path.join(Input, txt.split(',')[0])
                                image_source, image = load_image(img_path) # cropsize: 對原圖crop
                                m_image = cv2.imread(img_path)
                                boxes, logits, phrases = predict(
                                    model=model,
                                    image=image,
                                    caption=TEXT_PROMPT,
                                    box_threshold=BOX_TRESHOLD,
                                    text_threshold=TEXT_TRESHOLD)

                                bounding_boxes = bbox_convert(boxes, image_source)
                                nms_output = non_maximum_suppression_fast(bounding_boxes, overlapThresh=0.3)
                                tmp_image = m_image.copy()
                                for box in nms_output:
                                    box_int = box
                                    tmp = m_image[int(box_int[1]):int(box_int[3]), int(box_int[0]):int(box_int[2])]
                                    mask = np.ones_like(tmp) * 255
                                    tmp_image[int(box_int[1]):int(box_int[3]), int(box_int[0]):int(box_int[2])] = mask
                                objects = []
                                box_im, labels = [], []
                                for box in nms_output:
                                    box_int = box
                                    tmp = m_image[int(box_int[1]):int(box_int[3]), int(box_int[0]):int(box_int[2])]
                                    out_match = cv2.matchTemplate(tmp_image, tmp, cv2.TM_CCOEFF_NORMED)
                                    if len(np.where(out_match>0.97)[0]) == 0:
                                        save_box = [int(box_int[0]), int(box_int[2]), int(box_int[1]), int(box_int[3])]
                                        box_im.append([int(box_int[0]), int(box_int[1]), int(box_int[2]), int(box_int[3])])
                                        labels.append(label_name)
                                        objects.append({"name": label_name, "xmin": str(save_box[0]), "xmax": str(save_box[1]), "ymin": str(save_box[2]), "ymax": str(save_box[3])})
                                if len(objects) > 0:
                                    filename_text = img_path.split('\\')[-1]
                                    folder_text = img_path.split('\\')[-2]
                                    xml = create_xml(objects, filename_text, folder_text,  str(m_image.shape[1]), str(m_image.shape[0]), save_path = out_dir)
                                else:
                                    if n_file:
                                        filename_text = img_path.split('\\')[-1]
                                        folder_text = img_path.split('\\')[-2]
                                        create_non_xml(objects, filename_text, folder_text,  str(m_image.shape[1]), str(m_image.shape[0]), save_path = out_dir)
                        os.remove(txt_path)
                
                except Exception as e:
                    logger.error(txt_path0)
                    logger.error(e)
                    remove_file(tmp_path)
                    print(e)   
