import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))


from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes, draw_flow, draw_detector
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from utils.progress import Progress
from stabilization.stabilizer import Stabilizer
from counter.counter import Counter, Box, Line


class Sharingan(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        self.backSub = cv2.createBackgroundSubtractorKNN()

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        assert os.path.isfile(self.video_path), "Path error"
        self.vdo.open(self.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, self.args.output_name + ".avi")
            self.save_results_path = os.path.join(self.args.save_path, self.args.output_name + ".txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        # calculate the stabilization transform
        stable_fixer = Stabilizer(self.args.stable_period, Progress(0, 10))
        fixed_transform = stable_fixer.get_transform(self.vdo)
        progress = Progress(10, 99)

        # initialize detection line
        detection_line = Line(*self.args.detector_line.split(","))
        detection_counter = Counter(
            self.vdo.get(cv2.CAP_PROP_FPS),
            detection_line
        )
        width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        results = []
        idx_frame = 0
        self.vdo.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue
            if idx_frame >= len(fixed_transform):
                break

            start = time.time()

            # fetch image
            _, ori_im = self.vdo.retrieve()

            # fix image. stabilize then foreground masking
            fixed_im = stable_fixer.fix_frame(ori_im, fixed_transform[idx_frame], width, height)
            
            """
            fgMaskRaw = self.backSub.apply(fixed_im)
            fgMask = np.full(fgMaskRaw.shape, False, dtype=bool)
            """

            # a pixel become permeable iff (-5, +5) is foreground
            """
            fgMaskRaw = fgMaskRaw > 0.8
            for w in range(-5, 6):
                fgMaskHorz = np.roll(fgMaskRaw, w, axis=0)
                for h in range(-5, 6):
                    fgMaskRolled = np.roll(fgMaskHorz, h, axis=1)
                    fgMask |= fgMaskRolled
            """

            # mask out background
            """
            fgMask = np.stack([fgMask, fgMask, fgMask], axis=2)
            fg_im = np.multiply(fixed_im, fgMask)
            """
            fg_im = fixed_im
            
            # convert to rgb
            fg_im_rgb = cv2.cvtColor(fg_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(fg_im_rgb)

            # select traffic class
            mask = False
            mask |= cls_ids == 2 # car
            mask |= cls_ids == 5 # bus
            mask |= cls_ids == 7 # truck

            bbox_xywh = bbox_xywh[mask]
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, fg_im_rgb)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]

                for i in range(len(outputs)):
                    bb_xyxy, bb_id = bbox_xyxy[i], identities[i]
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                    detection_counter.update(bb_id, Box(*bb_xyxy))
                
                fg_im = draw_boxes(fg_im, bbox_xyxy, identities)
                results.append((idx_frame - 1, bbox_tlwh, identities))

            fg_im = draw_flow(fg_im, detection_counter.getFlow())
            fg_im = draw_detector(fg_im, detection_line)

            end = time.time()

            if self.args.display:
                cv2.imshow("test", fg_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(fg_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            log = "time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}, " \
                    .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs))
            log += progress.get_progress(idx_frame / len(fixed_transform))
            self.logger.info(log)
    
        print(f"Flow: {detection_counter.getFlow()}, " + Progress(99, 100).get_progress(100))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_mmdetection", type=str, default="./configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    # Sharingan specific parameters
    parser.add_argument("--detector_line", type=str, default='0,0,1000,1000')
    parser.add_argument("--stable_period", type=int, default=1000)
    parser.add_argument("--output_name", type=str, default='results')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    if args.mmdet:
        cfg.merge_from_file(args.config_mmdetection)
        cfg.USE_MMDET = True
    else:
        cfg.merge_from_file(args.config_detection)
        cfg.USE_MMDET = False
    cfg.merge_from_file(args.config_deepsort)
    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    with Sharingan(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
