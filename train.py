import os
import torch

from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()  # 构造L1损失函数
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    # print("dataloader=", dataloader)  # <torch.utils.data.dataloader.DataLoader object at 0x0000023BE5971AF0>
    max_iter = len(dataloader)  # 最大迭代次数是数据的长度,因为每次迭代四次
    # print("len(dataloader)=", max_iter)  # len(dataloader)= 526
    # 按需调整学习率，lr_steps是一个递增的list，gamma是学习率调整倍数，默认为0.1倍，这里根据参数设置为0.5，即下降50倍
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)

    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])  # 加载优化器的状态
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])  # 加载模型
        print('Resume from %d' % epoch)
        epoch += 1

    writer = SummaryWriter()  # 实例化摘要和文件
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')  # 周期时间
    iter_timer = Timer('m')  # 迭代时间
    best_psnr = -1

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)  # 将数据加载到指定的设备上

            optimizer.zero_grad()  # 将梯度置零
            pred_img = model(input_img)  # 将张量输入模型
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')  # 下采样
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')  # 下采样四倍
            l1 = criterion(pred_img[0], label_img4)  # 计算损失函数
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1 + l2 + l3  # 损失函数

            # signal_ndim=2 因为图像是二维的，normalized=False 说明不进行归一化，onesided=False 则是希望不要减少最后一个维度的大小
            label_fft1 = torch.rfft(label_img4, signal_ndim=2, normalized=False, onesided=False)
            pred_fft1 = torch.rfft(pred_img[0], signal_ndim=2, normalized=False, onesided=False)
            label_fft2 = torch.rfft(label_img2, signal_ndim=2, normalized=False, onesided=False)
            pred_fft2 = torch.rfft(pred_img[1], signal_ndim=2, normalized=False, onesided=False)
            label_fft3 = torch.rfft(label_img, signal_ndim=2, normalized=False, onesided=False)
            pred_fft3 = torch.rfft(pred_img[2], signal_ndim=2, normalized=False, onesided=False)

            f1 = criterion(pred_fft1, label_fft1)  # 经过傅里叶变换之后的Loss损失
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1 + f2 + f3

            loss = loss_content + 0.1 * loss_fft  # 总的loss损失,原来是0.1倍的loss_fft
            loss.backward()  # 计算当前张量的梯度
            optimizer.step()  # 更新所有的参数

            iter_pixel_adder(loss_content.item())  # 每次迭代之后
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())  # 内容损失
            epoch_fft_adder(loss_fft.item())  # 快速傅里叶变换损失
            # print("'iter_idx + 1'=", iter_idx + 1)

            if (iter_idx + 1) % args.print_freq == 0:  # 每100次迭代保存一次临时的model，显示一次
                lr = check_lr(optimizer)  # 检查lr
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(),
                                  iter_idx + (epoch_idx - 1) * max_iter)  # 计算像素损失，保存文件中供可视化使用
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()  # 每次迭代之后重置
                iter_fft_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:  # save_freq=100，每100个周期保存一次模型
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()  # 时间调度器计数
        if epoch_idx % args.valid_freq == 0:  # 每100个周期计算一次原始数据平均的PSNR
            val_gopro = _valid(model, args, epoch_idx)  # 计算一下原始数据集的平均峰值信噪比
            print("val_gopro==", val_gopro)  # 已修改
            print("epoch_idx==", epoch_idx)  # 已修改
            print('%03d epoch \n Average GOPRO PSNR %.2f dB' % (epoch_idx, val_gopro))  # 100个周期的平均峰值信噪比
            writer.add_scalar('PSNR_GOPRO', val_gopro, epoch_idx)  # 将所需的数据保存在文件里进行可视化，用来画图
            if val_gopro >= best_psnr:  # 平均的峰值信噪比大于-1,就可以保存模型
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
