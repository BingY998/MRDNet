import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)  # args.data_dir=dataset/GOPRO
    model.eval()
    psnr_adder = Adder()  # 实例化一个类(对象)

    with torch.no_grad():
        print('Start GoPro Evaluation')  # 对原始数据集进行评估
        for idx, data in enumerate(gopro):  # 枚举数据
            input_img, label_img = data
            input_img = input_img.to(device)
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)

            pred_clip = torch.clamp(pred[2], 0, 1)  # 截取图像
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)  # 计算PSNR
            #         print("psnr==", psnr)  # 已修改

            psnr_adder(psnr)
            print('\r%03d' % idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()  # 计算平均PSNR
