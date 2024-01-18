import os
import cv2
import time
import torch
import argparse
import openpyxl
import numpy as np
import glob
import natsort

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls
# from sklearn.metrics import confusion_matrix
from openpyxl import load_workbook

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG


# 유치장 실사 영상 데이터 로드, 파일명 정렬
source = glob.glob('./Data/prison_data/NewPrisonClip60/*')
source = natsort.natsorted(source)

# 학습된 ST-GCN 모델 로드, 저장 폴더 지정(ret/)
model_test_path = './Models/TSSTG/tsstg-model-best-1.pth'
out_path = './ret/'

# 엑셀에 예측 결과 저장 (y_preds) 
headers = ['Video', 'Frame', 'Prediction']
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.append(headers)

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                    kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

def create_folder(path):
    """
    Creates a new folder at the specified path if it doesn't already exist.
    """
    try:
        # Check if the folder already exists
        if not os.path.exists(path):
            os.makedirs(path)
            return f"Folder created at: {path}"
        else:
            return f"Folder already exists at: {path}"
    except Exception as e:
        return f"An error occurred: {e}"


#for k in range(31, len(source)): # 비낙상 for문 30건
for k in range(len(source)): # 낙상 for 문 30건
    source[k] = source[k].replace('\\', '/')
    video_path = out_path + source[k].split('/')[-1]
    create_folder(video_path)
    
    action_name = "pending.."
    if __name__ == '__main__':
        par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
        par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                            help='Source of camera or video file path.')
        par.add_argument('--detection_input_size', type=int, default=768, # default=384
                            help='Size of input in detection model in square must be divisible by 32 (int).')
        par.add_argument('--pose_input_size', type=str, default='288x160', # default=224x160
                            help='Size of input in pose model must be divisible by 32 (h, w)')
        par.add_argument('--pose_backbone', type=str, default='resnet50',
                            help='Backbone model for SPPE FastPose model.')
        par.add_argument('--show_detected', default=False, action='store_true',
                            help='Show all bounding box from detection.')
        par.add_argument('--show_skeleton', default=True, action='store_true',
                            help='Show skeleton pose.')
        par.add_argument('--save_out', type=str, default=video_path,
                            help='Save display to video file.')
        par.add_argument('--device', type=str, default='cuda',
                            help='Device to run model on cpu or cuda.')
        args = par.parse_args()

        device = args.device

        # DETECTION MODEL.
        inp_dets = args.detection_input_size
        detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

        # POSE MODEL.
        inp_pose = args.pose_input_size.split('x')
        inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
        pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

        # Tracker. (유치장 최대 인원 설정)
        max_age = 5
        tracker = Tracker(max_age=max_age, n_init=3)

        # Actions Estimate.
        action_model = TSSTG(weight_file=model_test_path)

        resize_fn = ResizePadding(inp_dets, inp_dets)

        cam_source = args.camera[k]
        print(cam_source)
        if type(cam_source) is str and os.path.isfile(cam_source):
            # Use loader thread with Q for video file.
            cam = CamLoader_Q(cam_source, queue_size=100000, preprocess=preproc).start()
        else:
            # Use normal thread loader for webcam.
            cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                            preprocess=preproc).start()

        outvid = False
        if args.save_out != '':
            outvid = True
            codec = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

        fps_time = 0
        f = 0
        while cam.grabbed():
            f += 1
            frame = cam.getitem()
            image = frame.copy()

            # Detect humans bbox in the frame with detector model.
            detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

            # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
            tracker.predict()
            # Merge two source of predicted bbox together.
            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det

            detections = []  # List of Detections object for tracking.
            if detected is not None:
                #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
                # Predict skeleton pose of each bboxs.
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

                # Create Detections object.
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                        ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in poses]

                # VISUALIZE.
                if args.show_detected:
                    for bb in detected[:, 0:5]:
                        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

            # Update tracks by matching each track information of current and previous frame or
            # create a new track if no matched.
            tracker.update(detections)

            # Predict Actions of each track.
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)

                action = 'pending..'
                clr = (0, 255, 0)
                
                # Use 15 frames time-steps to prediction. (예측 시 반영되는 프레임 개수)
                if len(track.keypoints_list) == 15:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    print(action_name)
                    if action_name == 'Fall Down':
                        clr = (255, 0, 0)

                # VISUALIZE.
                if track.time_since_update == 0:
                    if args.show_skeleton:
                        frame = draw_single(frame, track.keypoints_list[-1])
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                    frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, (255, 0, 0), 2)
                    frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, clr, 1)

            # Show Frame, 해상도 설정 등
            frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
            frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frame = frame[:, :, ::-1]
            fps_time = time.time()

            # 낙상 프레임 저장
            if outvid:
                writer.write(frame)
                cv2.imwrite(video_path + "/" + str(f).zfill(4) + ".jpg", frame)

            # 낙상 프레임 출력
            #cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 모델이 예측한 데이터 개수 기록
            if action_name == 'pending..':
                l_ret = -1
            elif action_name == "Fall Down":
                l_ret = 1
            else:
                l_ret = 0
            res = [source[k].split('/')[-1], f, l_ret]
            sheet.append(res)
            
        workbook.save(out_path + "pred_count.xlsx")
        
        # Clear resource.
        cam.stop()
        if outvid:
            writer.release()
        cv2.destroyAllWindows()
        
        
        