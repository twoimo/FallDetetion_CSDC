# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile

import cv2
import os
import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction
from mmengine.utils import track_iter_progress

from mmaction.apis import (detection_inference, inference_recognizer,
                           init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract
# from mmaction.apis.inference import save_data_to_file

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('--video', default = 'test_data/1_18_1.mp4', help='video file/url')
    parser.add_argument('--out_filename', default='demo/1_18_1_out.mp4', help='output filename')
    parser.add_argument(
        '--det-config',
        default='demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='the category id for human detection')
    parser.add_argument(
        '--pose-config',
        default='demo/demo_configs/'
        'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def visualize(args, frames, data_samples):
    global vid
    pose_config = mmengine.Config.fromfile(args.pose_config)
    visualizer = VISUALIZERS.build(pose_config.visualizer)
    visualizer.set_dataset_meta(data_samples[0].dataset_meta)

    vis_frames = []
    print('Drawing skeleton for each frame')
    for d, f in track_iter_progress(list(zip(data_samples, frames))):
        f = mmcv.imconvert(f, 'bgr', 'rgb')
        visualizer.add_datasample(
            'result',
            f,
            data_sample=d,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show=False,
            wait_time=0,
            out_file=None,
            kpt_thr=0.3)
        vis_frame = visualizer.get_image()
        vis_frames.append(vis_frame)

    vid = mpy.ImageSequenceClip(vis_frames, fps=30)
    vid.write_videofile(args.out_filename, remove_temp=True)


def main():
    global pose_results
    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, frames = frame_extract(args.video, tmp_dir.name)

    num_frame = len(frame_paths)
    h, w, _ = frames[0].shape

    # Get Human detection results.
    det_results, _ = detection_inference(args.det_config, args.det_checkpoint,
                                         frame_paths, args.det_score_thr,
                                         args.det_cat_id, args.device)
    torch.cuda.empty_cache()

    # Get Pose estimation results.
    pose_results, pose_data_samples = pose_inference(args.pose_config,
                                                     args.pose_checkpoint,
                                                     frame_paths, det_results,
                                                     args.device)
    torch.cuda.empty_cache()
    
    # save_data_to_file(det_results, 'demo/det_results.json')
    print(len(det_results))
    # print('det_results = ', det_results)
    # print(len(det_results))
    print(len(pose_results[0]))
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x['keypoints']) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_frame, num_person, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_frame, num_person, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        keypoint[i] = poses['keypoints']
        keypoint_score[i] = poses['keypoint_scores']

    fake_anno['keypoint'] = keypoint.transpose((1, 0, 2, 3))
    fake_anno['keypoint_score'] = keypoint_score.transpose((1, 0, 2))

    visualize(args, frames, pose_data_samples)
    
    # # pose_results 텍스트 파일 저장
    # save_data_to_file(pose_results, 'demo/pose_results.json')
    # print(len(pose_results))

    tmp_dir.cleanup()
    
if __name__ == '__main__':
    main()


def save_frames_as_images(video_path, output_path):
    # 비디오 파일 열기
    video = cv2.VideoCapture(video_path)
    
    # 프레임 수와 FPS 가져오기
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # 프레임을 저장할 폴더 생성
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    except OSError:
        print ('Error: Creating directory. ' +  output_path)

    # 프레임별로 이미지 저장
    for i in range(frame_count):
        # 프레임 읽기
        ret, frame = video.read()
        if not ret:
            break
        
        # 이미지 저장
        image_path = f"{output_path}/frame_{i}.jpg"
        cv2.imwrite(image_path, frame)
        print(f"Saved frame {i} as {image_path}")
    
    # 비디오 파일 닫기
    video.release()

# 입력 비디오 파일 경로
video_path = "./test_data/1_18_1.mp4"

# 이미지 저장 경로
output_path = "./1_18_1_out"

# 영상의 프레임을 이미지로 저장
save_frames_as_images(video_path, output_path)