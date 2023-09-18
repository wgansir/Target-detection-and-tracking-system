# 系统题目：目标检测跟踪一体化系统
# 作者：ChenHu
# Copyright： Free
# 更新时间： 20230802

# 重点逻辑：
# 当界面涉及到不断更新（大量数据）的时候，会由于对象多次创建出现数据累积的情况。目前替换更新策略走不通，目前走的是销毁重建策略。
# 具体而说，文本更新用的是替换更新策略，图片更新用的是销毁重建策略（替换更新出现界面上无图的现象）
# 注意点：不更新的数据避免多次创建，小数据多次更新要留意积累现象（次要），大数据多次更新要留意积累现象（主要）


import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
from PIL import Image, ImageTk
from yolo.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
import cv2
import random

from tkinter import filedialog, dialog
import torch
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import datetime
import os
import time
import argparse
import os
import platform
import sys
from pathlib import Path
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk, ImageDraw
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolo.models.common import DetectMultiBackend
from yolo.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolo.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolo.utils.plots import Annotator, colors, save_one_box
from yolo.utils.torch_utils import select_device, smart_inference_mode

def on_mouse_press(event):
    global start_x, start_y,is_waiting_for_draw
    start_x = event.x
    start_y = event.y


def on_mouse_release(event,img_input,img_orishape):
    global end_x, end_y, init_rect_draw
    global start_x, start_y
    global binding, binding_track, binding_track_release,is_waiting_for_draw
    global tracker
    global cropped_img


    end_x = event.x
    end_y = event.y

    # 创建一个临时画布
    # temp_canvas = tk.Canvas(window, width=img_input.width, height=img_input.height)
    # temp_canvas.place(relx=0.1, rely=0.1)

    # 在临时画布上绘制矩形框
    draw = ImageDraw.Draw(img_input)
    draw.rectangle((start_x, start_y, end_x, end_y), outline="red", width=2)

    # 将绘制后的图片转换为PhotoImage
    photo_with_rect = ImageTk.PhotoImage(img_input)

    # 在界面上更新图片
    imglabel_track_frame.configure(image=photo_with_rect)
    imglabel_track_frame.image = photo_with_rect
    window.update()

    # img_width = imglabel_track_frame.winfo_width()  # 注意label的长宽和图片长宽并不一致，我们应该计算与图片的相对坐标
    # img_height = imglabel_track_frame.winfo_height()

    img_width = img_input.width  # 显示图片本身长宽
    img_height = img_input.height

    im0_x = (start_x) * img_orishape.shape[1] / img_width
    im0_y = (start_y) * img_orishape.shape[0] / img_height

    # 计算点击事件相对于图片im0的坐标
    im0_x_2 = (event.x) * img_orishape.shape[1] / img_width
    im0_y_2 = (event.y) * img_orishape.shape[0] / img_height


    cropped_img = img_orishape[int(im0_y):int(im0_y_2), int(im0_x):int(im0_x_2)]
    init_rect_track = (im0_x, im0_y, im0_x_2 - im0_x, im0_y_2 - im0_y)
    tracker.init(img_orishape, init_rect_track)  # 跟踪器初始化

    imglabel_track_frame.unbind("<ButtonPress-1>", binding_track)
    imglabel_track_frame.unbind("<ButtonRelease-1>",binding_track_release)

    print('绘制结束')

    img_target = Image.fromarray(cropped_img[:,:,::-1])

    img_target = image_resize(img_target, 100, 100)

    img_target = ImageTk.PhotoImage(img_target)  # 用PIL模块的PhotoImage打开

    imglabel_target = tkinter.Label(window, bd=10, image=img_target)
    imglabel_target.image = img_target
    imglabel_target.place(relx=0.82, rely=0.13)
    window.update()

    is_waiting_for_draw = False




def handle_input(): # 输入帧数
    global frame_choose
    frame_choose = float(value_entry.get())
    print("User input:", frame_choose)

    frame_choose = frame_choose

#获得跟踪模型相关代码
import numpy as np



def det_and_track():
    global mode_choose

    mode_choose = "det_and_track"

    text_det_and_track = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="一体模式", fg="red")
    text_det_and_track.place(relx=0.045, rely=0.90)

def only_track():
    global mode_choose
    mode_choose = "only_track"
    text_only_track = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="跟踪模式", fg="red")
    text_only_track.place(relx=0.045, rely=0.90)

def crop_image_by_click(im0, xyxy_list, x, y):
    global init_rect
    # 将xyxy_list转换为NumPy数组
    xyxy_array = np.array(xyxy_list)

    # 检查每个目标框是否包含鼠标点击的位置
    for xyxy in xyxy_array:
        xmin, ymin, xmax, ymax = xyxy

        # 如果点击位置在目标框内，则裁剪图像并返回
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            cropped_img = im0[int(ymin):int(ymax), int(xmin):int(xmax)]
            init_rect = (xmin, ymin, xmax - xmin, ymax - ymin)
            return cropped_img

    # 如果点击位置不在任何目标框内，则返回提示信息
    return "点击区域无有效目标"


def wait_for_mouse_click():
    global is_waiting_for_click
    global init_rect

    is_waiting_for_click = True
    while is_waiting_for_click:
        window.update()


def wait_for_draw_target():
    global is_waiting_for_draw
    global init_rect


    while is_waiting_for_draw:
        window.update()

def handle_click(event,im0,xyxy_list):
    from PIL import Image, ImageTk
    global is_waiting_for_click
    global clicked_x, clicked_y
    global imglabel_track_frame
    global imglabel_target
    global img_target_flag
    global track_flag
    global template_img_choose
    global init_rect
    global prev_frame_time_2

    if img_target_flag==0:
        pass
    else:
        imglabel_target.destroy()
    # 点击事件：根据点击位置最上层的窗口计算坐标

    # # 计算点击事件相对于图片 im0 的坐标(相对总窗口window：未用)
    # img_x = imglabel_track_frame.winfo_x() # 相对左上角坐标
    # img_y = imglabel_track_frame.winfo_y()
    # img_width = imglabel_track_frame.winfo_width() # 本身长宽
    # img_height = imglabel_track_frame.winfo_height()
    #
    # # 计算点击事件相对于图片im0的坐标
    # im0_x = (event.x - img_x) * im0.shape[1] / img_width
    # im0_y = (event.y - img_y) * im0.shape[0] / img_height
    #


    # # 计算点击事件相对于图片 im0 的坐标(相对显式图片)
    img_width = imglabel_track_frame.winfo_width() # 显示图片本身长宽
    img_height = imglabel_track_frame.winfo_height()
    #
    # 计算点击事件相对于图片 im0 的坐标(相对显式图片)
    # img_width = im0.widch # 显示图片本身长宽
    # img_height = im0.height

    # 计算点击事件相对于图片im0的坐标 (有偏差但可接受，想改掉偏差需要改为img_width为界面上图片大小，而不是界面窗口大小)
    im0_x = (event.x) * im0.shape[1] / img_width
    im0_y = (event.y) * im0.shape[0] / img_height

    is_waiting_for_click = False

    cropped_img = crop_image_by_click(im0, xyxy_list, im0_x, im0_y)
    if isinstance(cropped_img, np.ndarray):
        # 显示裁剪后的图像


        img_target = Image.fromarray(cropped_img[:,:,::-1])

        img_target = image_resize(img_target,100,100)

        img_target = ImageTk.PhotoImage(img_target)  # 用PIL模块的PhotoImage打开

        imglabel_target = tkinter.Label(window, bd=10, image=img_target)
        imglabel_target.image = img_target
        imglabel_target.place(relx=0.82, rely=0.13)
        window.update()
        img_target_flag = 1

        track_flag = 1
        prev_frame_time_2 = time.time()

        window.unbind("<Button-1>", binding)

        template_img_choose  =  cropped_img

        tracker.init(im0, init_rect) # 跟踪器初始化

    else:
        if (event.x < img_width) and (event.y <img_height):
            messagebox.showerror("错误", "未选中目标")
            wait_for_mouse_click()

        else:
            messagebox.showerror("错误", "未启动跟踪模式")
            window.unbind("<Button-1>", binding)






class VideoPlayer:
    def __init__(self, master,video_path):
        # 创建主窗口
        self.master = master
        # self.master.geometry('640x480')

        # 创建标签用于显示视频帧
        self.label = tk.Label(self.master)
        self.label.pack()

        # 创建播放按钮
        self.play_button = tk.Button(self.master, text='Play', command=self.play_video)
        self.play_button.pack(pady=10)

        # 创建OpenCV视频对象
        self.cap = cv2.VideoCapture(video_path)

    def play_video(self,video_path):
        # 循环读取视频帧并显示在标签中
        while True:
            ret, frame = self.cap.read()
            if ret:
                # 将OpenCV图像转换为PIL图像并显示在标签中
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                photo = ImageTk.PhotoImage(image)
                self.label.configure(image=photo)
                self.label.image = photo

                # 等待一段时间
                self.master.after(30, lambda: None)
            else:
                break

        # 重置视频对象并重新播放
        self.cap.release()
        self.cap = cv2.VideoCapture(video_path)

    def run(self):
        self.master.mainloop()



def adjust_font_size(text):
    text_width = text.winfo_width()
    text_height = text.winfo_height()
    text_size = int(text['font'].split()[1])
    max_size = 50
    min_size = 5
    while (text.index('end-1c').split('.')[0] != 1 and
           (text.winfo_width() < text_width or text.winfo_height() < text_height) and
           text_size < max_size):
        text_size += 1
        text.configure(font=('Arial', text_size))
    while text_size > min_size and (text.winfo_width() > text_width or text.winfo_height() > text_height):
        text_size -= 1
        text.configure(font=('Arial', text_size))

def image_resize(img, screen_width=800, screen_height=500):
    image = img

    raw_width, raw_height = image.size[0], image.size[1]
    max_width, max_height = raw_width, screen_height
    min_width = max(raw_width, max_width)
    # 按照比例缩放
    min_height = int(raw_height * min_width / raw_width)
    # 第1次快速调整
    while min_height > screen_height:
        min_height = int(min_height * .9533)
    # 第2次精确微调
    while min_height < screen_height:
        min_height += 1
    # 按照比例缩放
    min_width = int(raw_width * min_height / raw_height)
    # 适应性调整
    while min_width > screen_width:
        min_width -= 1
    # 按照比例缩放
    min_height = int(raw_height * min_width / raw_width)
    return image.resize((min_width, min_height))


def open_file_output():
    from PIL import Image
    '''
    打开文件
    :return:local_
    '''
    global file_path
    global file_text
    global photo
    global img
    global cap_flag
    global first_choose
    global text_load_track, text_load_detect
    global track_flag
    global mode_choose

    if mode_choose != "det_and_track" and mode_choose != "only_track":
        messagebox.showwarning("Warning", "未选择跟踪模式")
        return


    track_flag = 0

    if first_choose == 1:
        first_choose = 0
    else:
        text_begin.destroy()
        text_end.destroy()
        # text_save.destroy()
        text_load_track.destroy()
        text_load_detect.destroy()


    cap_flag = 0
    file_path = filedialog.askopenfilename(title=u'选择视频')
    print('打开文件：', file_path)
    if file_path is not None:
        file_text = "文件路径为：" + file_path

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error opening video file")

    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)

    # 关闭视频文件
    cap.release()

    # img = Image.open(file_path)  # 打开图片
    # img = cv.imread(str(file_path))

    img = image_resize(img)
    photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开

    imglabel = tkinter.Label(window, bd=10, image=photo)
    imglabel.place(relx=0.1, rely=0.1)


def run_contral():
    global opt_detect

    main(opt_detect)





@smart_inference_mode()
def run_loadmodel(
        weights=ROOT / 'yolo/yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        save_video = False,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):

    from PIL import Image
    global cap_flag
    global text_begin
    global text_end
    global text_model_creat
    global frame_choose
    global track_flag
    global tracker
    global hp
    global model_detect
    global seen_detect, windows_detect, dt_detect,dataset_detect,webcam_detect,save_dir_detect,vid_path_detect, vid_writer_detect,names_detect

    track_flag = 0


    if cap_flag==1:



        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam_detect = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir_detect = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir_detect / 'labels' if save_txt else save_dir_detect).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model_detect = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names_detect, pt = model_detect.stride, model_detect.names, model_detect.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam_detect:
            view_img = check_imshow(warn=True)
            dataset_detect = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset_detect)
        elif screenshot:
            dataset_detect = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            # dataset_detect = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            dataset_detect = None
        vid_path_detect, vid_writer_detect = [None] * bs, [None] * bs

        # Run inference
        model_detect.warmup(imgsz=(1 if pt or model_detect.triton else bs, 3, *imgsz))  # warmup
        seen_detect, windows_detect, dt_detect = 0, [], (Profile(), Profile(), Profile())


    else:
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam_detect = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir_detect = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir_detect / 'labels' if save_txt else save_dir_detect).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model_detect = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names_detect, pt = model_detect.stride, model_detect.names, model_detect.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size


        # Dataloader
        bs = 1  # batch_size
        if webcam_detect:
            view_img = check_imshow(warn=True)
            dataset_detect = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset_detect)
        elif screenshot:
            dataset_detect = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset_detect = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path_detect, vid_writer_detect = [None] * bs, [None] * bs

        # Run inference
        model_detect.warmup(imgsz=(1 if pt or model_detect.triton else bs, 3, *imgsz))  # warmup
        seen_detect, windows_detect, dt_detect = 0, [], (Profile(), Profile(), Profile())






@smart_inference_mode()
def run(
        weights=ROOT / 'yolo/yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        save_video = False,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):

    from PIL import Image

    global cap_flag
    global text_begin
    global text_end
    global text_model_creat
    global frame_choose
    global track_flag
    global tracker
    global hp
    global model_detect
    global seen_detect, windows_detect, dt_detect,dataset_detect,webcam_detect,save_dir_detect,vid_path_detect, vid_writer_detect,names_detect
    global prev_frame_time
    global prev_frame_time_2
    global imglabel_track_frame
    global imglabel_dect_frame
    global img_target_flag
    global binding,binding_track,binding_track_release
    global template_img_choose
    global init_rect
    global cap
    import PIL
    global position_fps_title
    global position_fps
    global is_waiting_for_draw

    # 防止部件反复创建到界面（不会替换，而会累加，不同于普通变量）引起内存增加
    position_label_flag = True
    position_fps_flag = True
    imglabel_tracker_flag = True
    imglabel_dect_frame_flag = True
    position_time_dect_flag = True
    position_track_time_flag = True

    global mode_choose
    global init_rect_draw

    if mode_choose != "det_and_track" and mode_choose != "only_track":
        messagebox.showwarning("Warning", "未选择跟踪模式")
        return

    # 选择仅跟踪还是跟踪检测一体化
    if mode_choose == "only_track" :
        # 摄像头模式
        if cap_flag == 1:
            track_flag = 0

            prev_frame_time_2 = time.time()

            # text_model_creat.destroy()
            text_begin = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="开始处理", fg="red")
            text_begin.place(relx=0.465, rely=0.90)
            window.update()

            frame_count = 0
            fps_num = 1
            track_points = []

            # for path, im, im0s, vid_cap, s in dataset_detect:  # 遍历视频序列的每一帧（直接遍历图片，不预先遍历序列）

            while (cap.isOpened()):

                # if fps_num % 10 == 0 and fps_num!=0 :
                #     del ret_flag,img,im0s,img_track,im,im0
                #     position_time_title.destroy()
                #     position_time.destroy()
                #     position_fps.destroy()
                #     position_fps_title.destroy()

                fps_num = fps_num + 1
                frame_count = frame_count + 1

                ret_flag, img = cap.read()
                img_track = img  # 变色图片，但源代码用的是这个

                im0s = img[:, :, ::-1]  # 原图(恢复原色)
                im = im0s
                im0 = im0s.copy()


                prev_frame_time = time.time()


                if track_flag == 0:

                    im_vision = Image.fromarray(im0)
                    img = image_resize(im_vision)
                    photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
                    imglabel_track_frame = tkinter.Label(window, bd=10, image=photo)
                    imglabel_track_frame.image = photo
                    imglabel_track_frame.place(relx=0.1, rely=0.1)
                    window.update()

                if frame_count == int(frame_choose):
                    is_waiting_for_draw = True
                    track_flag = 1
                    print("绘制跟踪目标")
                    binding_track = imglabel_track_frame.bind("<ButtonPress-1>", on_mouse_press)
                    binding_track_release = imglabel_track_frame.bind("<ButtonRelease-1>", lambda event: on_mouse_release(event, img,img_track))
                    wait_for_draw_target()

                if track_flag == 1:

                    # 计算跟踪时间
                    if show_track_3.get():
                        current_time_2 = time.time()
                        frame_interval_2 = current_time_2 - prev_frame_time_2
                        # 更新上一帧时间戳为当前时间戳
                        prev_frame_time_2 = time.time()

                        if position_track_time_flag:
                            position_time_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                            position_time_title.place(relx=0.781, rely=0.77)
                            position_time = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                            position_time.place(relx=0.81, rely=0.77)
                            # 更新标签的文本内容

                            position_time_title.config(text=f"用时: ")
                            position_time.config(text=f"{frame_interval_2:<.4f} s")
                            window.update()

                            position_track_time_flag = False

                        else:
                            position_time_title.config(text=f"用时: ")
                            position_time.config(text=f"{frame_interval_2:<.4f} s")
                            window.update()

                    # template_img = template_img_choose
                    search_img = img_track

                    # 跟踪目标并获取边界框
                    outputs = tracker.track(search_img)
                    bbox = list(map(int, outputs['bbox']))

                    # 此处如果是原图则会出错，可理解为cv2默认格式是变色格式
                    cv2.rectangle(search_img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)

                    # 添加目标轨迹点
                    track_points.append((bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))
                    if len(track_points) > 80:  # 轨迹长度
                        track_points.pop(0)

                    # 绘制目标轨迹
                    if track_flag == 1 and show_track.get():
                        for i, point in enumerate(track_points):
                            size = int((i + 1) * 0.2)
                            if size >= 4:
                                size = 4
                            # color = (0, 255 - size * 20, 0)
                            color = (0, 0, 255)
                            cv2.circle(search_img, point, size, color, -1)

                    # 显式目标位置信息
                    if show_track_2.get() and position_label_flag:
                        position_label2 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_label2.place(relx=0.781, rely=0.62)
                        position_label3 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_label3.place(relx=0.781, rely=0.65)

                        xmin, ymin, xmax, ymax = bbox
                        position_label2.config(text=f"xmin: {xmin:<5d}  ymin: {ymin:<5d}")
                        position_label3.config(text=f"xmax: {xmax:<5d}  ymax: {ymax:<5d}")
                        window.update()

                        position_label_flag = False

                    if show_track_2.get() and (not position_label_flag):
                        # 更新标签的文本内容
                        xmin, ymin, xmax, ymax = bbox
                        position_label2.config(text=f"xmin: {xmin:<5d}  ymin: {ymin:<5d}")
                        position_label3.config(text=f"xmax: {xmax:<5d}  ymax: {ymax:<5d}")
                        window.update()

                    if imglabel_tracker_flag:

                        search_img = search_img[:, :, ::-1]
                        search_img = Image.fromarray(search_img)
                        search_img = image_resize(search_img)
                        search_img_photo = ImageTk.PhotoImage(search_img)  # 用PIL模块的PhotoImage打开
                        imglabel_tracker = tkinter.Label(window, bd=10, image=search_img_photo)
                        imglabel_tracker.image = search_img_photo
                        imglabel_tracker.place(relx=0.1, rely=0.1)
                        window.update()

                        imglabel_tracker_flag = False
                    else:
                        imglabel_tracker.destroy()

                        search_img = search_img[:, :, ::-1]
                        search_img = Image.fromarray(search_img)
                        search_img = image_resize(search_img)
                        search_img_photo = ImageTk.PhotoImage(search_img)  # 用PIL模块的PhotoImage打开

                        imglabel_tracker = tkinter.Label(window, bd=10, image=search_img_photo)
                        imglabel_tracker.image = search_img
                        imglabel_tracker.place(relx=0.1, rely=0.1)
                        window.update()

                # 显式帧数
                if position_fps_flag:
                    position_fps_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                    position_fps_title.place(relx=0.781, rely=0.8)
                    position_fps = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                    position_fps.place(relx=0.81, rely=0.8)
                    position_fps_title.config(text=f"帧数: ")
                    position_fps.config(text=f"{fps_num:<5d} 帧")
                    window.update()
                    position_fps_flag = False
                # 更新标签的文本内容

                else:

                    position_fps.config(text=f"{fps_num:<5d} 帧")
                    window.update()


        # 不调用摄像头
        else:

            track_flag = 0

            # text_model_creat.destroy()
            text_begin = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="开始处理", fg="red")
            text_begin.place(relx=0.465, rely=0.90)
            window.update()

            frame_count = 0

            track_points = []

            for path, im, im0s, vid_cap, s in dataset_detect:  # 遍历视频序列的每一帧（直接遍历图片，不预先遍历序列）

                prev_frame_time = time.time()
                if track_flag == 0:  # 未进入跟踪（只检测）

                    frame_count = frame_count + 1
                    with dt_detect[0]:
                        im = torch.from_numpy(im).to(model_detect.device)
                        im = im.half() if model_detect.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # Inference
                    with dt_detect[1]:
                        visualize = increment_path(save_dir_detect / Path(path).stem,
                                                   mkdir=True) if visualize else False
                        pred = model_detect(im, augment=augment, visualize=visualize)  # 输出结果

                    # NMS
                    with dt_detect[2]:
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Process predictions

                    for i, det in enumerate(pred):  # 遍历每帧的输出结果（目标）
                        seen_detect += 1

                        if webcam_detect:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset_detect.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset_detect, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir_detect / p.name)  # im.jpg
                        txt_path = str(save_dir_detect / 'labels' / p.stem) + (
                            '' if dataset_detect.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names_detect))
                        normal_fps = 1
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, 5].unique():
                                n = (det[:, 5] == c).sum()  # detections per class
                                s += f"{n} {names_detect[int(c)]}{'s' * (n > 1)}, "  # add to string

                            xyxy_list = []
                            for *xyxy, conf, cls in reversed(det):

                                #
                                if int(cls) not in [0]:  # 修改类别
                                    continue

                                xyxy_list.append(np.array([tensor.cpu().numpy() for tensor in xyxy]))

                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (
                                        names_detect[c] if hide_conf else f'{names_detect[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                if save_crop:
                                    save_one_box(xyxy, imc,
                                                 file=save_dir_detect / 'crops' / names_detect[c] / f'{p.stem}.jpg',
                                                 BGR=True)

                        # Stream results
                        im0 = annotator.result()

                        # change
                        if track_flag == 0:

                            if show_track_3.get():
                                current_time = time.time()
                                frame_interval = current_time - prev_frame_time
                                # 更新上一帧时间戳为当前时间戳
                                prev_frame_time = time.time()

                                position_time_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"),
                                                                    text="")
                                position_time_title.place(relx=0.781, rely=0.77)
                                position_time = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                                position_time.place(relx=0.81, rely=0.77)
                                # 更新标签的文本内容

                                position_time_title.config(text=f"用时: ")
                                position_time.config(text=f"{frame_interval:<.4f} s")
                                window.update()

                            im_vision = im0[:, :, ::-1]
                            im_vision = Image.fromarray(im_vision)
                            img = image_resize(im_vision)
                            photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
                            imglabel_track_frame = tkinter.Label(window, bd=10, image=photo)
                            imglabel_track_frame.image = photo
                            imglabel_track_frame.place(relx=0.1, rely=0.1)
                            window.update()
                            # time.sleep(0.08)

                    if frame_count == int(frame_choose):
                        img_target_flag = 0
                        binding = window.bind("<Button-1>",
                                              lambda event: handle_click(event, img_track, xyxy_list))  # 绑定点击事件
                        print("选择跟踪目标")
                        wait_for_mouse_click()

                    # Print time (inference-only)
                    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt_detect[1].dt * 1E3:.1f}ms")

                    # Print results
                    t = tuple(x.t / seen_detect * 1E3 for x in dt_detect)  # speeds per image
                    LOGGER.info(
                        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
                    if save_txt or save_img:
                        s = f"\n{len(list(save_dir_detect.glob('labels/*.txt')))} labels saved to {save_dir_detect / 'labels'}" if save_txt else ''
                        LOGGER.info(f"Results saved to {colorstr('bold', save_dir_detect)}{s}")
                    if update:
                        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

                if track_flag == 1:

                    # 计算跟踪时间
                    if show_track_3.get():
                        current_time_2 = time.time()
                        frame_interval_2 = current_time_2 - prev_frame_time_2
                        # 更新上一帧时间戳为当前时间戳
                        prev_frame_time_2 = time.time()

                        position_time_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_time_title.place(relx=0.781, rely=0.77)
                        position_time = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_time.place(relx=0.81, rely=0.77)
                        # 更新标签的文本内容

                        position_time_title.config(text=f"用时: ")
                        position_time.config(text=f"{frame_interval_2:<.4f} s")
                        window.update()

                    template_img = template_img_choose
                    search_img = im0s

                    # 跟踪目标并获取边界框
                    outputs = tracker.track(search_img)
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(search_img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)

                    # 添加目标轨迹点
                    track_points.append((bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))
                    if len(track_points) > 80:  # 轨迹长度
                        track_points.pop(0)

                    # 绘制目标轨迹
                    if track_flag == 1 and show_track.get():
                        for i, point in enumerate(track_points):
                            size = int((i + 1) * 0.2)
                            if size >= 4:
                                size = 4
                            # color = (0, 255 - size * 20, 0)
                            color = (0, 0, 255)
                            cv2.circle(search_img, point, size, color, -1)

                    # 显式目标位置信息
                    if track_flag == 1 and show_track_2.get():
                        position_label2 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_label2.place(relx=0.781, rely=0.62)
                        position_label3 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_label3.place(relx=0.781, rely=0.65)
                        # 更新标签的文本内容
                        xmin, ymin, xmax, ymax = bbox
                        position_label2.config(text=f"xmin: {xmin:<5d}  ymin: {ymin:<5d}")
                        position_label3.config(text=f"xmax: {xmax:<5d}  ymax: {ymax:<5d}")

                    search_img = search_img[:, :, ::-1]
                    search_img = Image.fromarray(search_img)
                    search_img = image_resize(search_img)
                    search_img_photo = ImageTk.PhotoImage(search_img)  # 用PIL模块的PhotoImage打开
                    imglabel_tracker = tkinter.Label(window, bd=10, image=search_img_photo)
                    imglabel_tracker.image = search_img_photo
                    imglabel_tracker.place(relx=0.1, rely=0.1)
                    window.update()

    else:

        # 调用摄像头
        if cap_flag==1:


            if show_track_online.get():

                track_flag = 0
                trans_track_first = True

                # text_model_creat.destroy()
                text_begin = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="开始处理", fg="red")
                text_begin.place(relx=0.465, rely=0.90)
                window.update()

                frame_count = 0
                fps_num = 1
                track_points = []

                # for path, im, im0s, vid_cap, s in dataset_detect:  # 遍历视频序列的每一帧（直接遍历图片，不预先遍历序列）

                have_target_count  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                while (cap.isOpened()):

                    # if fps_num % 10 == 0 and fps_num!=0 :
                    #     del ret_flag,img,im0s,img_track,im,im0
                    #     position_time_title.destroy()
                    #     position_time.destroy()
                    #     position_fps.destroy()
                    #     position_fps_title.destroy()
                    have_target_flag = False

                    prev_frame_time_2 = time.time()

                    fps_num = fps_num + 1

                    ret_flag, img = cap.read()
                    img_track = img # 变色图片，但源代码用的是这个

                    im0s = img[:, :, ::-1] # 原图(恢复原色)
                    im = im0s
                    im0 = im0s.copy()

                    im = letterbox(im, [640,640], stride=32, auto=True)[0]  # padded resize
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = np.ascontiguousarray(im)  # contiguous
                    # 备注：源检测代码的im是否是变色图片变换后的有待考证，此处是原图变换后的




                    prev_frame_time = time.time()
                    if track_flag == 0:  # 未进入跟踪（只检测）

                        frame_count = frame_count + 1
                        with dt_detect[0]:
                            im = torch.from_numpy(im).to(model_detect.device)
                            im = im.half() if model_detect.fp16 else im.float()  # uint8 to fp16/32
                            im /= 255  # 0 - 255 to 0.0 - 1.0
                            if len(im.shape) == 3:
                                im = im[None]  # expand for batch dim

                        # Inference
                        with dt_detect[1]:
                            # visualize = increment_path(save_dir_detect / Path(path).stem, mkdir=True) if visualize else False
                            pred = model_detect(im, augment=augment, visualize=visualize)  # 输出结果

                        # NMS
                        with dt_detect[2]:
                            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                        # Second-stage classifier (optional)
                        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                        # Process predictions

                        for i, det in enumerate(pred):  # 遍历每帧的输出结果（目标）
                            seen_detect += 1


                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            imc = im0.copy() if save_crop else im0  # for save_crop
                            annotator = Annotator(im0, line_width=line_thickness, example=str(names_detect))
                            normal_fps = 1
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                # for c in det[:, 5].unique():
                                #     n = (det[:, 5] == c).sum()  # detections per class
                                #     s += f"{n} {names_detect[int(c)]}{'s' * (n > 1)}, "  # add to string

                                xyxy_list = []
                                for *xyxy, conf, cls in reversed(det):

                                    if int(cls) not in [0]:  # 修改类别
                                        continue
                                    else:
                                        have_target_flag = True

                                    xyxy_list.append(np.array([tensor.cpu().numpy() for tensor in xyxy]))

                                    # if save_txt:  # Write to file
                                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    #         -1).tolist()  # normalized xywh
                                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    #     with open(f'{txt_path}.txt', 'a') as f:
                                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                    if save_img or save_crop or view_img:  # Add bbox to image
                                        c = int(cls)  # integer class
                                        label = None if hide_labels else (
                                            names_detect[c] if hide_conf else f'{names_detect[c]} {conf:.2f}')
                                        annotator.box_label(xyxy, label, color=colors(c, True))
                                    if save_crop:
                                        save_one_box(xyxy, imc,
                                                     file=save_dir_detect / 'crops' / names_detect[c] / f'{p.stem}.jpg',
                                                     BGR=True)

                            # Stream results
                            im0 = annotator.result()




                            # change
                            if track_flag == 0:

                                if show_track_3.get():
                                    current_time = time.time()
                                    frame_interval = current_time - prev_frame_time
                                    # 更新上一帧时间戳为当前时间戳
                                    prev_frame_time = time.time()

                                    if position_time_dect_flag:
                                        position_time_title_dect = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                                        position_time_title_dect.place(relx=0.781, rely=0.77)
                                        position_time_dect = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                                        position_time_dect.place(relx=0.81, rely=0.77)

                                        position_time_title_dect.config(text=f"用时: ")
                                        position_time_dect.config(text=f"{frame_interval:<.4f} s")
                                        window.update()

                                        position_time_dect_flag = False
                                        # 更新标签的文本内容
                                    else:

                                        position_time_dect.config(text=f"{frame_interval:<.4f} s")
                                        window.update()

                                # if fps_num > 2:
                                #     imglabel_track_frame.destroy()

                                if imglabel_dect_frame_flag:
                                    im_vision = im0
                                    im_vision = Image.fromarray(im_vision)
                                    img = image_resize(im_vision)
                                    photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
                                    imglabel_track_frame = tkinter.Label(window, bd=10, image=photo)
                                    imglabel_track_frame.image = photo
                                    imglabel_track_frame.place(relx=0.1, rely=0.1)
                                    window.update()

                                    imglabel_dect_frame_flag = False

                                else:
                                    imglabel_track_frame.destroy()

                                    im_vision = im0
                                    im_vision = Image.fromarray(im_vision)
                                    img = image_resize(im_vision)
                                    photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
                                    imglabel_track_frame = tkinter.Label(window, bd=10, image=photo)
                                    imglabel_track_frame.image = photo
                                    imglabel_track_frame.place(relx=0.1, rely=0.1)
                                    window.update()


                        if update:
                            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

                    # 此处写跟踪代码

                    if have_target_flag:
                        have_target_count.append(1)
                        if len(have_target_count) > 10:
                            have_target_count.pop(0)
                    else:
                        have_target_count.append(0)
                        if len(have_target_count) > 10:
                            have_target_count.pop(0)

                    if have_target_count.count(1) >= 5 and trans_track_first:
                        trans_track_first = False
                        track_flag = 1

                        xyxy = random.choice(xyxy_list)
                        xmin, ymin, xmax, ymax = xyxy

                        cropped_img = img_track[int(ymin):int(ymax), int(xmin):int(xmax)]
                        init_rect = (xmin, ymin, xmax - xmin, ymax - ymin)

                        img_target = Image.fromarray(cropped_img[:, :, ::-1])
                        img_target = image_resize(img_target, 100, 100)
                        img_target = ImageTk.PhotoImage(img_target)  # 用PIL模块的PhotoImage打开

                        imglabel_target = tkinter.Label(window, bd=10, image=img_target)
                        imglabel_target.image = img_target
                        imglabel_target.place(relx=0.82, rely=0.13)
                        window.update()
                        img_target_flag = 1

                        # template_img_choose =

                        tracker.init(img_track, init_rect)  # 跟踪器初始化




                    if track_flag == 1:

                        # 计算跟踪时间
                        if show_track_3.get():
                            current_time_2 = time.time()
                            frame_interval_2 = current_time_2 - prev_frame_time_2
                            # 更新上一帧时间戳为当前时间戳
                            prev_frame_time_2 = time.time()

                            if position_track_time_flag:
                                position_time_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                                position_time_title.place(relx=0.781, rely=0.77)
                                position_time = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                                position_time.place(relx=0.81, rely=0.77)
                                # 更新标签的文本内容

                                position_time_title.config(text=f"用时: ")
                                position_time.config(text=f"{frame_interval_2:<.4f} s")
                                window.update()

                                position_track_time_flag = False

                            else:
                                position_time_title.config(text=f"用时: ")
                                position_time.config(text=f"{frame_interval_2:<.4f} s")
                                window.update()

                        # template_img = template_img_choose
                        search_img = img_track

                        # 跟踪目标并获取边界框
                        outputs = tracker.track(search_img)
                        bbox = list(map(int, outputs['bbox']))

                        # 此处如果是原图则会出错，可理解为cv2默认格式是变色格式
                        cv2.rectangle(search_img, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 3)

                        # 添加目标轨迹点
                        track_points.append((bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))
                        if len(track_points) > 80:  # 轨迹长度
                            track_points.pop(0)

                        # 绘制目标轨迹
                        if track_flag == 1 and show_track.get():
                            for i, point in enumerate(track_points):
                                size = int((i + 1) * 0.2)
                                if size >= 4:
                                    size = 4
                                # color = (0, 255 - size * 20, 0)
                                color = (0, 0, 255)
                                cv2.circle(search_img, point, size, color, -1)

                        # 显式目标位置信息
                        if  show_track_2.get() and position_label_flag :
                            position_label2 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                            position_label2.place(relx=0.781, rely=0.62)
                            position_label3 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                            position_label3.place(relx=0.781, rely=0.65)

                            xmin, ymin, xmax, ymax = bbox
                            position_label2.config(text=f"xmin: {xmin:<5d}  ymin: {ymin:<5d}")
                            position_label3.config(text=f"xmax: {xmax:<5d}  ymax: {ymax:<5d}")
                            window.update()

                            position_label_flag = False

                        if  show_track_2.get() and (not position_label_flag) :
                            # 更新标签的文本内容
                            xmin, ymin, xmax, ymax = bbox
                            position_label2.config(text=f"xmin: {xmin:<5d}  ymin: {ymin:<5d}")
                            position_label3.config(text=f"xmax: {xmax:<5d}  ymax: {ymax:<5d}")
                            window.update()

                        if imglabel_tracker_flag:

                            search_img = search_img[:, :, ::-1]
                            search_img = Image.fromarray(search_img)
                            search_img = image_resize(search_img)
                            search_img_photo = ImageTk.PhotoImage(search_img)  # 用PIL模块的PhotoImage打开
                            imglabel_tracker = tkinter.Label(window, bd=10, image = search_img_photo)
                            imglabel_tracker.image = search_img_photo
                            imglabel_tracker.place(relx=0.1, rely=0.1)
                            window.update()

                            imglabel_tracker_flag = False
                        else:
                            imglabel_tracker.destroy()

                            search_img = search_img[:, :, ::-1]
                            search_img = Image.fromarray(search_img)
                            search_img = image_resize(search_img)
                            search_img_photo = ImageTk.PhotoImage(search_img)  # 用PIL模块的PhotoImage打开

                            imglabel_tracker = tkinter.Label(window, bd=10, image=search_img_photo)
                            imglabel_tracker.image = search_img
                            imglabel_tracker.place(relx=0.1, rely=0.1)
                            window.update()




                    # 显式帧数
                    if position_fps_flag:
                        position_fps_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_fps_title.place(relx=0.781, rely=0.8)
                        position_fps = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_fps.place(relx=0.81, rely=0.8)
                        position_fps_title.config(text=f"帧数: ")
                        position_fps.config(text=f"{fps_num:<5d} 帧")
                        window.update()
                        position_fps_flag = False
                    # 更新标签的文本内容

                    else:

                        position_fps.config(text=f"{fps_num:<5d} 帧")
                        window.update()

            else:
                track_flag = 0

                # text_model_creat.destroy()
                text_begin = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="开始处理", fg="red")
                text_begin.place(relx=0.465, rely=0.90)
                window.update()

                frame_count = 0
                fps_num = 1
                track_points = []

                # for path, im, im0s, vid_cap, s in dataset_detect:  # 遍历视频序列的每一帧（直接遍历图片，不预先遍历序列）

                while (cap.isOpened()):

                    # if fps_num % 10 == 0 and fps_num!=0 :
                    #     del ret_flag,img,im0s,img_track,im,im0
                    #     position_time_title.destroy()
                    #     position_time.destroy()
                    #     position_fps.destroy()
                    #     position_fps_title.destroy()

                    fps_num = fps_num + 1

                    ret_flag, img = cap.read()
                    img_track = img  # 变色图片，但源代码用的是这个

                    im0s = img[:, :, ::-1]  # 原图(恢复原色)
                    im = im0s
                    im0 = im0s.copy()

                    im = letterbox(im, [640, 640], stride=32, auto=True)[0]  # padded resize
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = np.ascontiguousarray(im)  # contiguous
                    # 备注：源检测代码的im是否是变色图片变换后的有待考证，此处是原图变换后的

                    prev_frame_time = time.time()
                    if track_flag == 0:  # 未进入跟踪（只检测）

                        frame_count = frame_count + 1
                        with dt_detect[0]:
                            im = torch.from_numpy(im).to(model_detect.device)
                            im = im.half() if model_detect.fp16 else im.float()  # uint8 to fp16/32
                            im /= 255  # 0 - 255 to 0.0 - 1.0
                            if len(im.shape) == 3:
                                im = im[None]  # expand for batch dim

                        # Inference
                        with dt_detect[1]:
                            # visualize = increment_path(save_dir_detect / Path(path).stem, mkdir=True) if visualize else False
                            pred = model_detect(im, augment=augment, visualize=visualize)  # 输出结果

                        # NMS
                        with dt_detect[2]:
                            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                                       max_det=max_det)

                        # Second-stage classifier (optional)
                        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                        # Process predictions

                        for i, det in enumerate(pred):  # 遍历每帧的输出结果（目标）
                            seen_detect += 1

                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            imc = im0.copy() if save_crop else im0  # for save_crop
                            annotator = Annotator(im0, line_width=line_thickness, example=str(names_detect))
                            normal_fps = 1
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                # for c in det[:, 5].unique():
                                #     n = (det[:, 5] == c).sum()  # detections per class
                                #     s += f"{n} {names_detect[int(c)]}{'s' * (n > 1)}, "  # add to string

                                xyxy_list = []
                                for *xyxy, conf, cls in reversed(det):

                                    if int(cls) not in [0]:  # 修改类别
                                        continue

                                    xyxy_list.append(np.array([tensor.cpu().numpy() for tensor in xyxy]))

                                    # if save_txt:  # Write to file
                                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    #         -1).tolist()  # normalized xywh
                                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    #     with open(f'{txt_path}.txt', 'a') as f:
                                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                    if save_img or save_crop or view_img:  # Add bbox to image
                                        c = int(cls)  # integer class
                                        label = None if hide_labels else (
                                            names_detect[c] if hide_conf else f'{names_detect[c]} {conf:.2f}')
                                        annotator.box_label(xyxy, label, color=colors(c, True))
                                    if save_crop:
                                        save_one_box(xyxy, imc,
                                                     file=save_dir_detect / 'crops' / names_detect[c] / f'{p.stem}.jpg',
                                                     BGR=True)

                            # Stream results
                            im0 = annotator.result()

                            # change
                            if track_flag == 0:

                                if show_track_3.get():
                                    current_time = time.time()
                                    frame_interval = current_time - prev_frame_time
                                    # 更新上一帧时间戳为当前时间戳
                                    prev_frame_time = time.time()

                                    if position_time_dect_flag:
                                        position_time_title_dect = tkinter.Label(window,
                                                                                 font=("Microsoft YaHei", 11, "bold"),
                                                                                 text="")
                                        position_time_title_dect.place(relx=0.781, rely=0.77)
                                        position_time_dect = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"),
                                                                           text="")
                                        position_time_dect.place(relx=0.81, rely=0.77)

                                        position_time_title_dect.config(text=f"用时: ")
                                        position_time_dect.config(text=f"{frame_interval:<.4f} s")
                                        window.update()

                                        position_time_dect_flag = False
                                        # 更新标签的文本内容
                                    else:

                                        position_time_dect.config(text=f"{frame_interval:<.4f} s")
                                        window.update()

                                # if fps_num > 2:
                                #     imglabel_track_frame.destroy()

                                if imglabel_dect_frame_flag:
                                    im_vision = im0
                                    im_vision = Image.fromarray(im_vision)
                                    img = image_resize(im_vision)
                                    photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
                                    imglabel_track_frame = tkinter.Label(window, bd=10, image=photo)
                                    imglabel_track_frame.image = photo
                                    imglabel_track_frame.place(relx=0.1, rely=0.1)
                                    window.update()

                                    imglabel_dect_frame_flag = False

                                else:
                                    imglabel_track_frame.destroy()

                                    im_vision = im0
                                    im_vision = Image.fromarray(im_vision)
                                    img = image_resize(im_vision)
                                    photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
                                    imglabel_track_frame = tkinter.Label(window, bd=10, image=photo)
                                    imglabel_track_frame.image = photo
                                    imglabel_track_frame.place(relx=0.1, rely=0.1)
                                    window.update()

                        if frame_count == int(frame_choose):
                            # global img_target_flag
                            # global binding
                            # global template_img_choose
                            # global init_rect

                            img_target_flag = 0
                            binding = window.bind("<Button-1>",
                                                  lambda event: handle_click(event, img_track, xyxy_list))  # 绑定点击事件
                            print("选择跟踪目标")
                            wait_for_mouse_click()

                        if update:
                            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

                    # 此处写跟踪代码

                    if track_flag == 1:

                        # 计算跟踪时间
                        if show_track_3.get():
                            current_time_2 = time.time()
                            frame_interval_2 = current_time_2 - prev_frame_time_2
                            # 更新上一帧时间戳为当前时间戳
                            prev_frame_time_2 = time.time()

                            if position_track_time_flag:
                                position_time_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"),
                                                                    text="")
                                position_time_title.place(relx=0.781, rely=0.77)
                                position_time = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                                position_time.place(relx=0.81, rely=0.77)
                                # 更新标签的文本内容

                                position_time_title.config(text=f"用时: ")
                                position_time.config(text=f"{frame_interval_2:<.4f} s")
                                window.update()

                                position_track_time_flag = False

                            else:
                                position_time_title.config(text=f"用时: ")
                                position_time.config(text=f"{frame_interval_2:<.4f} s")
                                window.update()

                        template_img = template_img_choose
                        search_img = img_track

                        # 跟踪目标并获取边界框
                        outputs = tracker.track(search_img)
                        bbox = list(map(int, outputs['bbox']))

                        # 此处如果是原图则会出错，可理解为cv2默认格式是变色格式
                        cv2.rectangle(search_img, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 3)

                        # 添加目标轨迹点
                        track_points.append((bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))
                        if len(track_points) > 80:  # 轨迹长度
                            track_points.pop(0)

                        # 绘制目标轨迹
                        if track_flag == 1 and show_track.get():
                            for i, point in enumerate(track_points):
                                size = int((i + 1) * 0.2)
                                if size >= 4:
                                    size = 4
                                # color = (0, 255 - size * 20, 0)
                                color = (0, 0, 255)
                                cv2.circle(search_img, point, size, color, -1)

                        # 显式目标位置信息
                        if show_track_2.get() and position_label_flag:
                            position_label2 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                            position_label2.place(relx=0.781, rely=0.62)
                            position_label3 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                            position_label3.place(relx=0.781, rely=0.65)

                            xmin, ymin, xmax, ymax = bbox
                            position_label2.config(text=f"xmin: {xmin:<5d}  ymin: {ymin:<5d}")
                            position_label3.config(text=f"xmax: {xmax:<5d}  ymax: {ymax:<5d}")
                            window.update()

                            position_label_flag = False

                        if show_track_2.get() and (not position_label_flag):
                            # 更新标签的文本内容
                            xmin, ymin, xmax, ymax = bbox
                            position_label2.config(text=f"xmin: {xmin:<5d}  ymin: {ymin:<5d}")
                            position_label3.config(text=f"xmax: {xmax:<5d}  ymax: {ymax:<5d}")
                            window.update()

                        if imglabel_tracker_flag:

                            search_img = search_img[:, :, ::-1]
                            search_img = Image.fromarray(search_img)
                            search_img = image_resize(search_img)
                            search_img_photo = ImageTk.PhotoImage(search_img)  # 用PIL模块的PhotoImage打开
                            imglabel_tracker = tkinter.Label(window, bd=10, image=search_img_photo)
                            imglabel_tracker.image = search_img_photo
                            imglabel_tracker.place(relx=0.1, rely=0.1)
                            window.update()

                            imglabel_tracker_flag = False
                        else:
                            imglabel_tracker.destroy()

                            search_img = search_img[:, :, ::-1]
                            search_img = Image.fromarray(search_img)
                            search_img = image_resize(search_img)
                            search_img_photo = ImageTk.PhotoImage(search_img)  # 用PIL模块的PhotoImage打开

                            imglabel_tracker = tkinter.Label(window, bd=10, image=search_img_photo)
                            imglabel_tracker.image = search_img
                            imglabel_tracker.place(relx=0.1, rely=0.1)
                            window.update()

                    # 显式帧数
                    if position_fps_flag:
                        position_fps_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_fps_title.place(relx=0.781, rely=0.8)
                        position_fps = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_fps.place(relx=0.81, rely=0.8)
                        position_fps_title.config(text=f"帧数: ")
                        position_fps.config(text=f"{fps_num:<5d} 帧")
                        window.update()
                        position_fps_flag = False
                    # 更新标签的文本内容

                    else:

                        position_fps.config(text=f"{fps_num:<5d} 帧")
                        window.update()


        # 不调用摄像头
        else:

            track_flag = 0

            # text_model_creat.destroy()
            text_begin = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="开始处理", fg="red")
            text_begin.place(relx=0.465, rely=0.90)
            window.update()

            frame_count = 0

            track_points = []

            for path, im, im0s, vid_cap, s in dataset_detect: # 遍历视频序列的每一帧（直接遍历图片，不预先遍历序列）

                prev_frame_time = time.time()
                if track_flag == 0 : #未进入跟踪（只检测）

                    frame_count = frame_count + 1
                    with dt_detect[0]:
                        im = torch.from_numpy(im).to(model_detect.device)
                        im = im.half() if model_detect.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # Inference
                    with dt_detect[1]:
                        visualize = increment_path(save_dir_detect / Path(path).stem, mkdir=True) if visualize else False
                        pred = model_detect(im, augment=augment, visualize=visualize) # 输出结果

                    # NMS
                    with dt_detect[2]:
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Process predictions


                    for i, det in enumerate(pred):  # 遍历每帧的输出结果（目标）
                        seen_detect += 1

                        if webcam_detect:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset_detect.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset_detect, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir_detect / p.name)  # im.jpg
                        txt_path = str(save_dir_detect / 'labels' / p.stem) + ('' if dataset_detect.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names_detect))
                        normal_fps = 1
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, 5].unique():
                                n = (det[:, 5] == c).sum()  # detections per class
                                s += f"{n} {names_detect[int(c)]}{'s' * (n > 1)}, "  # add to string

                            xyxy_list = []
                            for *xyxy, conf, cls in reversed(det):


                                #
                                if int(cls) not in [0]: # 修改类别
                                    continue

                                xyxy_list.append(np.array([tensor.cpu().numpy() for tensor in xyxy]))

                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (names_detect[c] if hide_conf else f'{names_detect[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                if save_crop:
                                    save_one_box(xyxy, imc, file=save_dir_detect / 'crops' / names_detect[c] / f'{p.stem}.jpg', BGR=True)


                        # Stream results
                        im0 = annotator.result()

                        # change
                        if track_flag == 0:

                            if show_track_3.get():
                                current_time = time.time()
                                frame_interval = current_time - prev_frame_time
                                # 更新上一帧时间戳为当前时间戳
                                prev_frame_time = time.time()

                                position_time_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                                position_time_title.place(relx=0.781, rely=0.77)
                                position_time = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                                position_time.place(relx=0.81, rely=0.77)
                                # 更新标签的文本内容

                                position_time_title.config(text=f"用时: ")
                                position_time.config(text=f"{frame_interval:<.4f} s")
                                window.update()

                            im_vision = im0[:, :, ::-1]
                            im_vision = Image.fromarray(im_vision)
                            img = image_resize(im_vision)
                            photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
                            imglabel_track_frame = tkinter.Label(window, bd=10, image=photo)
                            imglabel_track_frame.image = photo
                            imglabel_track_frame.place(relx=0.1, rely=0.1)
                            window.update()
                            # time.sleep(0.08)



                    if frame_count==int(frame_choose):

                        img_target_flag = 0
                        binding = window.bind("<Button-1>", lambda event: handle_click(event,img_track,xyxy_list)) # 绑定点击事件
                        print("选择跟踪目标")
                        wait_for_mouse_click()

                    # Print time (inference-only)
                    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt_detect[1].dt * 1E3:.1f}ms")

                    # Print results
                    t = tuple(x.t / seen_detect * 1E3 for x in dt_detect)  # speeds per image
                    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
                    if save_txt or save_img:
                        s = f"\n{len(list(save_dir_detect.glob('labels/*.txt')))} labels saved to {save_dir_detect / 'labels'}" if save_txt else ''
                        LOGGER.info(f"Results saved to {colorstr('bold', save_dir_detect)}{s}")
                    if update:
                        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)



                if track_flag == 1:

                    # 计算跟踪时间
                    if show_track_3.get():
                        current_time_2 = time.time()
                        frame_interval_2 = current_time_2 - prev_frame_time_2
                        # 更新上一帧时间戳为当前时间戳
                        prev_frame_time_2 = time.time()

                        position_time_title = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_time_title.place(relx=0.781, rely=0.77)
                        position_time = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_time.place(relx=0.81, rely=0.77)
                        # 更新标签的文本内容

                        position_time_title.config(text=f"用时: ")
                        position_time.config(text=f"{frame_interval_2:<.4f} s")
                        window.update()


                    template_img = template_img_choose
                    search_img = im0s

                    # 跟踪目标并获取边界框
                    outputs = tracker.track(search_img)
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(search_img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)

                    # 添加目标轨迹点
                    track_points.append((bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))
                    if len(track_points) > 80: # 轨迹长度
                        track_points.pop(0)

                    # 绘制目标轨迹
                    if track_flag == 1 and show_track.get():
                        for i, point in enumerate(track_points):
                            size = int((i + 1)*0.2)
                            if size>=4:
                                size = 4
                            # color = (0, 255 - size * 20, 0)
                            color = (0, 0, 255)
                            cv2.circle(search_img, point, size, color, -1)

                    # 显式目标位置信息
                    if track_flag == 1 and show_track_2.get():
                        position_label2 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_label2.place(relx=0.781, rely=0.62)
                        position_label3 = tkinter.Label(window, font=("Microsoft YaHei", 11, "bold"), text="")
                        position_label3.place(relx=0.781, rely=0.65)
                        # 更新标签的文本内容
                        xmin, ymin, xmax, ymax = bbox
                        position_label2.config(text=f"xmin: {xmin:<5d}  ymin: {ymin:<5d}")
                        position_label3.config(text=f"xmax: {xmax:<5d}  ymax: {ymax:<5d}")


                    search_img = search_img[:, :, ::-1]
                    search_img = Image.fromarray(search_img)
                    search_img = image_resize(search_img)
                    search_img_photo = ImageTk.PhotoImage(search_img)  # 用PIL模块的PhotoImage打开
                    imglabel_tracker = tkinter.Label(window, bd=10, image=search_img_photo)
                    imglabel_tracker.image = search_img_photo
                    imglabel_tracker.place(relx=0.1, rely=0.1)
                    window.update()




    text_end = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="视频处理结束", fg="red")
    text_end.place(relx=0.365, rely=0.90)






def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolo/yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default="", help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default=False, help='do not save images/videos') # 是否保存
    parser.add_argument('--save_video', default=False, help='do not save images/videos')  # 是否保存

    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main_loadmodel(opt):
    global cap_flag
    # check_requirements(exclude=('tensorboard', 'thop'))
    if cap_flag==1:
        # opt.source = " "
        run_loadmodel(**vars(opt))
    else:
        opt.source = file_path
        run_loadmodel(**vars(opt))


def main(opt):
    global file_path
    global cap_flag
    # check_requirements(exclude=('tensorboard', 'thop'))
    if cap_flag == 1:
        # opt.source = file_path
        run(**vars(opt))
    else:
        opt.source = file_path
        run(**vars(opt))



def save_img():
    global text_save
    text_save = tkinter.Label(window, bd=10, font=("Arial", 15, "bold"), text="视频保存成功", fg="red")
    text_save.place(relx=0.512, rely=0.85)




def get_from_camera():
    global img_camera
    global cap_flag
    import PIL
    global cap
    global mode_choose

    if mode_choose != "det_and_track" and mode_choose != "only_track":
        messagebox.showwarning("Warning", "未选择跟踪模式")
        return

    cap_flag = 1
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret_flag, img = cap.read()
        img_camera = img[:,:,::-1]

        img_input = PIL.Image.fromarray(img_camera)

        img_input = image_resize(img_input)
        photo = ImageTk.PhotoImage(img_input)

        imglabel = tkinter.Label(window, bd=10, image=photo)
        imglabel.image = photo  # 保持引用，避免被垃圾回收器回收  !!!!此处非常重要
        imglabel.place(relx=0.1, rely=0.1)

        window.update()
        break



def load_model():
    global tracker
    global hp
    global opt_detect
    global text_load_track,text_load_detect

    import os
    import argparse

    import cv2
    import torch
    import numpy as np
    from glob import glob

    from siamrpnpp.pysot.core.config import cfg
    from siamrpnpp.pysot.models.model_builder import ModelBuilder
    from siamrpnpp.pysot.tracker.tracker_builder import build_tracker

    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, default="siamrpnpp/experiments/siamrpn_r50_l234_dwxcorr/config.yaml",
                        help='config file')
    parser.add_argument('--snapshot', type=str, default="siamrpnpp/experiments/siamrpn_r50_l234_dwxcorr/model.pth",
                        help='model name')
    parser.add_argument('--video_name', default='demo/bag.avi', type=str,
                        help='videos or image files')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)


    # ----------------------------------------------



    text_load_track = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="跟踪加载结束", fg="red")
    text_load_track.place(relx=0.325, rely=0.90)

    # 检测模型相关
    opt_detect = parse_opt()
    main_loadmodel(opt_detect)

    text_load_detect = tkinter.Label(window, bd=10, font=("Microsoft YaHei", 15, "bold"), text="检测加载结束", fg="red")
    text_load_detect.place(relx=0.325, rely=0.945)

def destroy_system():
    global cap
    #
    # cap.release()
    # cv2.destroyAllWindows()
    window.destroy()


if __name__ == '__main__':
    from PIL import ImageTk, Image

    window = tkinter.Tk()
    window.title('目标检测跟踪系统')
    window.geometry('1300x800')

    header_frame = tk.Frame(window, bg='purple', height=100)
    header_frame.pack(fill=tk.X)
    header_label = tk.Label(header_frame, text='目标检测跟踪系统', font=("Microsoft YaHei", 20, "bold"), fg='white', bg='purple')
    header_label.pack(expand=True)

    # # 界面右下角放置校徽图片
    # # 加载图片
    # image_path = './data/school_label.png'
    # image = Image.open(image_path)
    # image = image.resize((100, 100))  # 调整图片大小
    # photo = ImageTk.PhotoImage(image)
    # # 创建标签并显示图片
    # image_label = tk.Label(window, image=photo)
    # image_label.image = photo
    # image_label.place(relx=1, rely=1, anchor='se')

    # # 设置背景
    # canvas = tk.Canvas(window, width=window.winfo_screenwidth(), height=window.winfo_screenheight())
    # image = tk.PhotoImage(file="./data/img_back_2.png")
    # # 将图片放置于 Canvas 控件中
    # canvas.create_image(0, 0, image=image, anchor="nw")
    # # 将 Canvas 控件放置于窗口中
    # canvas.pack()

    # 模块1
    label_frame_txt = tk.Label(window, text="选定目标样式", font=("Microsoft YaHei", 15, "bold"))
    label_frame_txt.place(relx=0.8, rely=0.06)
    label_frame = tk.LabelFrame(window, text="", font=("Microsoft YaHei", 15, "bold"))
    label_frame.place(relx=0.773, rely=0.1, relwidth=0.15, relheight=0.2)

    # 模块2
    label_frame_txt = tk.Label(window, text="相关设置", font=("Microsoft YaHei", 15, "bold"))
    label_frame_txt.place(relx=0.81, rely=0.31)
    label_frame = tk.LabelFrame(window, text="", font=("Microsoft YaHei", 15, "bold"))
    label_frame.place(relx=0.773, rely=0.35, relwidth=0.15, relheight=0.2)

    # 模块2
    label_frame_txt = tk.Label(window, text="目标位置", font=("Microsoft YaHei", 15, "bold"))
    label_frame_txt.place(relx=0.81, rely=0.56)
    label_frame = tk.LabelFrame(window, text="", font=("Microsoft YaHei", 15, "bold"))
    label_frame.place(relx=0.773, rely=0.6, relwidth=0.15, relheight=0.1)

    # 模块3
    label_frame_txt = tk.Label(window, text="其他数据", font=("Microsoft YaHei", 15, "bold"))
    label_frame_txt.place(relx=0.81, rely=0.71)
    label_frame = tk.LabelFrame(window, text="", font=("Microsoft YaHei", 15, "bold"))
    label_frame.place(relx=0.773, rely=0.75, relwidth=0.15, relheight=0.1)

    # 跟踪模型选择
    label_track_model_name = tk.Label(window, text="跟踪模型：", font=("Microsoft YaHei", 15, "bold"))
    label_track_model_name.place(relx=0.775, rely=0.855)
    label_track_model_name_1 = tk.Label(window, text="SiamRPN++", font=("Microsoft YaHei", 15, "bold"),fg="red")
    label_track_model_name_1.place(relx=0.845, rely=0.855)

    window.update()





    first_choose = 1 # 清空提示字符的标识符
    frame_choose = 1 # 跟踪首帧（默认1）




    value_entry = tkinter.Entry(window)
    value_entry.pack()
    value_entry.place(relx=0.9, rely=0.4, anchor='se')
    button = tkinter.Button(window, text="跟踪首帧设置", command=handle_input)
    button.pack()
    button.place(relx=0.88, rely=0.45, anchor='se')


    button0_1 = tkinter.Button(window, text='一体化模式', command=det_and_track, width=15, height=2)  # 加括号会自动执行（！！）
    button0_2 = tkinter.Button(window, text='仅跟踪模式', command=only_track, width=15, height=2)  # 加括号会自动执行（！！）

    button1 = tkinter.Button(window, text='选择测试视频', command=open_file_output, width=15, height=2)  # 加括号会自动执行（！！）
    button1_2 = tkinter.Button(window, text='调用摄像头', command=get_from_camera, width=15, height=2)  # 加括号会自动执行（！！）
    # button5 = tkinter.Button(window, text='退出',bg = "red",fg = "white", command=lambda: window.destroy(), width=10, height=2)
    button5 = tkinter.Button(window, text='退出',bg = "red",fg = "white", command=destroy_system, width=10, height=2)
    button3 = tkinter.Button(window, text='处理', command=run_contral, width=10, height=2)  # 加括号会自动执行（！！）
    button2 = tkinter.Button(window, text='加载模型', command=load_model, width=10, height=2)
    # button4 = tkinter.Button(window, text='结果保存本地', command=save_img, width=10, height=2)

    button0_1.place(relx=0.13, rely=0.9, anchor='se')  # 相对位置，放置按钮
    button0_2.place(relx=0.13, rely=0.83, anchor='se')

    button1.place(relx=0.28, rely=0.83, anchor='se')  # 相对位置，放置按钮
    button1_2.place(relx=0.28, rely=0.9, anchor='se')


    button2.place(relx=0.41, rely=0.9, anchor='se')  # 相对位置，放置按钮
    button3.place(relx=0.535, rely=0.9, anchor='se')  # 相对位置，放置按钮
    # button4.place(relx=0.8, rely=0.85, anchor='se')  # 相对位置，放置按钮
    button5.place(relx=0.66, rely=0.9, anchor='se')  # 相对位置，放置按钮


    show_track = tk.BooleanVar() # "显示轨迹"
    show_track.set(True)  # 默认勾选
    # 创建一个勾选按钮
    check_button = tk.Checkbutton(window, text="显示轨迹", variable=show_track)
    # 将勾选按钮放置在合适的位置
    check_button.config(width=6, height=2)
    check_button.place(relx=0.78, rely=0.45)

    show_track_2 = tk.BooleanVar() # "显示目标位置"
    show_track_2.set(True)  # 默认勾选
    # 创建一个勾选按钮
    check_button_2 = tk.Checkbutton(window, text="目标位置", variable=show_track_2)
    # 将勾选按钮放置在合适的位置
    check_button_2.config(width=6, height=2)
    check_button_2.place(relx=0.85, rely=0.45)

    show_track_online = tk.BooleanVar()  # "在线检测跟踪"
    show_track_online.set(False)  # 默认勾选
    # 创建一个勾选按钮
    show_track_online_button = tk.Checkbutton(window, text="在线", variable=show_track_online)
    # 将勾选按钮放置在合适的位置
    show_track_online_button.config(width=6, height=2)
    show_track_online_button.place(relx=0.84, rely=0.49)

    show_track_3 = tk.BooleanVar()  # "显示时间用时"
    show_track_3.set(True)  # 默认勾选
    # 创建一个勾选按钮
    check_button_3 = tk.Checkbutton(window, text="显式时间", variable=show_track_3)
    # 将勾选按钮放置在合适的位置
    check_button_3.config(width=6, height=2)
    check_button_3.place(relx=0.78, rely=0.49)

    mode_choose = "default"

window.mainloop()




