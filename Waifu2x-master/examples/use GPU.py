from torchvision.utils import save_image
import time
from utils.prepare_images import *

lr = "./benchmark/miku_small.png"
def DCSCN_infer():
    pre_trained = "../model_check_points/DCSCN/DCSCN_weights_387epos_L12_noise_1.pt"
    model = DCSCN(color_channel=3,
                  up_scale=2,
                  feature_layers=12,
                  first_feature_filters=196,
                  last_feature_filters=48,
                  reconstruction_filters=128,
                  up_sampler_filters=32)

    model.load_state_dict(torch.load(pre_trained, map_location='cpu'))

    st = time.time()
    img = Image.open(lr).convert("RGB")
    img_up = img.resize((2 * img.size[0], 2 * img.size[1]), Image.BILINEAR)
    print(time.time()-st)
    img = to_tensor(img).unsqueeze(0)
    img_up = to_tensor(img_up).unsqueeze(0)

    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()
        img_up = img_up.cuda()
    print(time.time()-st)
    out = model((img, img_up))
    print(time.time()-st)
    save_image(out, './benchmark/miku_dcscn.png')

def CRAN_V2_infer():
    model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                            single_conv_size=3, single_conv_group=1,
                            scale=2, activation=nn.LeakyReLU(0.1),
                            SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))

    # model_cran_v2 = network_to_half(model_cran_v2)
    checkpoint = "../model_check_points/CRAN_V2/CARN_model_checkpoint.pt"
    model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
    # if use GPU, then comment out the next line so it can use fp16.
    # model_cran_v2 = model_cran_v2.float()


    st = time.time()
    img = Image.open(lr).convert("RGB")
    img_t = to_tensor(img).unsqueeze(0)
    # used to compare the origin
    img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.BICUBIC)

    # overlapping split
    # if input image is too large, then split it into overlapped patches
    # details can be found at [here](https://github.com/nagadomi/waifu2x/issues/238)
    img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    print(time.time()-st) # 0.65
    with torch.no_grad():
        out = [model_cran_v2(i) for i in img_patches]
    img_upscale = img_splitter.merge_img_tensor(out)
    print(time.time()-st)  # 3.7
    # final = torch.cat([img_t, img_upscale])
    # save_image(final, './benchmark/CRAN_hr.png', nrow=2)
    save_image(img_upscale, './benchmark/CRAN_hr.png', nrow=2)

if __name__ == '__main__':
    # CRAN_V2_infer()  # 3.7s
    DCSCN_infer()  # 3.5s
    # m,n,d=2,4,3
    # WQ=WK=WV=[[0.5]*d]*n
    # WQ = np.array(WQ)
    # WK = np.array(WK)
    # WV = np.array(WV)
    # A = np.array([[1,2,3,4],[5,10,5,7]])
    # wij = np.dot(K,Q.T)
    # def softmax(x, axis=1):
    #     row_max = x.max(axis=axis)
    #     row_max=row_max.reshape(-1, 1)
    #     x = x - row_max
    #     x_exp = np.exp(x)
    #     x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    #     s = x_exp / x_sum
    #     return s
    # np.dot(softmax(wij),V)
