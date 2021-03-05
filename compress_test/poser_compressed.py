import os
import sys

sys.path.append(os.getcwd())

import time
import numpy as np
import PIL.Image
import PIL.ImageTk
import cv2
import torch
import dlib

from poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from puppet.head_pose_solver import HeadPoseSolver
from puppet.util import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
    compute_mouth_normalized_ratio
from tha.combiner import CombinerSpec
from tha.face_morpher import FaceMorpherSpec
from tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from util import rgba_to_numpy_image, extract_pytorch_image_from_filelike

import torch.onnx


cuda = torch.device('cuda')

class Puppet_Core():
    def __init__(self):
        self.torch_device = cuda
        self.poser = MorphRotateCombinePoser256Param6(
            morph_module_spec=FaceMorpherSpec(),
            morph_module_file_name="E:/work/pycharm_v2/talking-head-anime-landing/data/face_morpher.pt",
            rotate_module_spec=TwoAlgoFaceRotatorSpec(),
            rotate_module_file_name="E:/work/pycharm_v2/talking-head-anime-landing/data/two_algo_face_rotator.pt",
            combine_module_spec=CombinerSpec(),
            combine_module_file_name="E:/work/pycharm_v2/talking-head-anime-landing/data/combiner.pt",
            device=cuda)
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_locator = dlib.shape_predictor("E:/work/pycharm_v2/talking-head-anime-landing/data/shape_predictor_68_face_landmarks.dat")
        self.head_pose_solver = HeadPoseSolver()
        self.pose_size = len(self.poser.pose_parameters())

    def run(self,source_img,frame,save_path):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(rgb_frame)
        euler_angles = None
        face_landmarks = None
        if len(faces) > 0:
            face_rect = faces[0]
            face_landmarks = self.landmark_locator(rgb_frame, face_rect)
            face_box_points, euler_angles = self.head_pose_solver.solve_head_pose(face_landmarks)

            if euler_angles is not None and source_img is not None:
                self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
                self.current_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
                self.current_pose[1] = max(min(-euler_angles.item(1) / 15.0, 1.0), -1.0)
                self.current_pose[2] = max(min(euler_angles.item(2) / 15.0, 1.0), -1.0)

                # if self.last_pose is None:
                # self.last_pose = self.current_pose
                # else:
                #     self.current_pose = self.current_pose * 0.5 + self.last_pose * 0.5  # smoothing
                #     self.last_pose = self.current_pose

                eye_min_ratio = 0.15
                eye_max_ratio = 0.25
                left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio)
                self.current_pose[3] = 1 - left_eye_normalized_ratio
                right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks,
                                                                                eye_min_ratio,
                                                                                eye_max_ratio)
                self.current_pose[4] = 1 - right_eye_normalized_ratio

                min_mouth_ratio = 0.02
                max_mouth_ratio = 0.3
                mouth_normalized_ratio = compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio)
                self.current_pose[5] = mouth_normalized_ratio
                self.current_pose = self.current_pose.unsqueeze(dim=0)

                st = time.time()
                posed_image = self.poser.pose(source_image=source_img, pose=self.current_pose).detach().cpu()
                print("Core Time(poser.pose): ", time.time()-st)
                numpy_image = rgba_to_numpy_image(posed_image[0])
                pil_image = PIL.Image.fromarray(np.uint8(np.rint(numpy_image * 255.0)), mode='RGBA')
                pil_image.save(save_path)
                # TODO Core of demo, compress this
        return

if __name__ == '__main__':
    demo = Puppet_Core()

    img_file = r"E:\work\pycharm_v2\talking-head-anime-landing\data/illust/waifu_00_256.png"
    source_img= extract_pytorch_image_from_filelike(img_file).to(cuda).unsqueeze(dim=0)
    save_file = "../save_img.png"
    frame = cv2.imread(r"E:\work\pycharm_v2\talking-head-anime-landing\my.png")

    start_time = time.time()
    demo.run(source_img,frame,save_file)
    print("Total Run Time: ",time.time()-start_time)


    # import torchvision.models as models
    #
    # resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    # import torch
    # BATCH_SIZE = 64
    # dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
    # torch.onnx.export(resnext50_32x4d, dummy_input, "resnet50_onnx_model.onnx", verbose=False)
