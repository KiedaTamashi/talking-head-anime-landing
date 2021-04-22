from SRutils.Models import *
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import time
import PIL

class SuperResolutionModule:
    def __init__(self,pre_trained = "Waifu2x/model_check_points/DCSCN/DCSCN_weights_387epos_L12_noise_1.pt"):
        self.model = DCSCN(color_channel=3,up_scale=2, feature_layers=12, first_feature_filters=196,
                        last_feature_filters=48, reconstruction_filters=128, up_sampler_filters=32)
        self.model.load_state_dict(torch.load(pre_trained, map_location='cpu'))
        self.model = self.model.cuda()
        self.unloader = ToPILImage()

    def inference(self,lr): # img must be 256x256
        # img = Image.open(lr).convert("RGB")
        img = lr.convert("RGB")
        img_up = img.resize((2 * img.size[0], 2 * img.size[1]), Image.BILINEAR)
        img = to_tensor(img).unsqueeze(0)
        img_up = to_tensor(img_up).unsqueeze(0)
        if torch.cuda.is_available():
            # model = self.model.cuda()
            img = img.cuda()
            img_up = img_up.cuda()
        else:
            self.model = self.model.cpu()
        # print(time.time()-st)
        out = self.model((img, img_up))
        out = out.cpu().squeeze(0)
        out = self.unloader(out)
        return out
        # print(time.time()-st)
        # save_image(out, './benchmark/miku_dcscn.png')


def transparent_back(img):
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((0,0))
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:# or color_0[:-1] in [(255,0,0),(0,255,0),(0,0,255)]:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
    return img