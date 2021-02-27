import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import PIL.Image
import PIL.ImageTk
import cv2
import torch

from poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from puppet.head_pose_solver import HeadPoseSolver
from poser.poser import Poser
from puppet.util import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
    compute_mouth_normalized_ratio
from tha.combiner import CombinerSpec
from tha.face_morpher import FaceMorpherSpec
from tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from util import rgba_to_numpy_image, extract_pytorch_image_from_filelike


cuda = torch.device('cuda')
poser = MorphRotateCombinePoser256Param6(
    morph_module_spec=FaceMorpherSpec(),
    morph_module_file_name="E:/work/pycharm_v2/talking-head-anime-landing/data/face_morpher.pt",
    rotate_module_spec=TwoAlgoFaceRotatorSpec(),
    rotate_module_file_name="E:/work/pycharm_v2/talking-head-anime-landing/data/two_algo_face_rotator.pt",
    combine_module_spec=CombinerSpec(),
    combine_module_file_name="E:/work/pycharm_v2/talking-head-anime-landing/data/combiner.pt",
    device=cuda)

img_file = r"data/illust/waifu_00_256.png"

source_img= extract_pytorch_image_from_filelike(img_file).to(cuda).unsqueeze(dim=0)
current_pose = None

poser.pose(source_image=source_img,pose=current_pose)