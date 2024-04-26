'''
test 
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

from srcs.model.infwide_modules_a import wiener_deblur_mc
from srcs.model.rl_filter import richardson_lucy

if __name__=='__main__':
    imgk = cv2.imread('dataset/NightShot_demo/gt/0001.jpg')
    # imgk = cv2.imread('dataset/NightShot_demo/testing_data_gt/02 (1).jpg')
    imgk = cv2.cvtColor(imgk, cv2.COLOR_BGR2RGB).astype(np.float32)/255

    # psfk = cv2.imread('dataset/NightShot_demo/kernel/psf02.png')
    psfk = cv2.imread('dataset/NightShot_demo/testkernel/00 (1).png')
    psfk = cv2.cvtColor(psfk, cv2.COLOR_BGR2GRAY)
    psfk = psfk.astype(np.float32)/np.sum(psfk)

    coded_blur_img = ndimage.convolve(
        imgk, np.expand_dims(psfk, axis=2), mode='wrap').astype(np.float32)


    # now change to torch
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    img_input = torch.from_numpy(imgk.transpose(2,0,1))


    x_input = torch.from_numpy(coded_blur_img.transpose(2,0,1))

    psf = np.expand_dims(np.float32(psfk), axis=2)
    psf = torch.from_numpy(psf.transpose(2,0,1))

    img_input = img_input.unsqueeze(0).to(device)
    x_input = x_input.unsqueeze(0).to(device)
    psf = psf.unsqueeze(0).to(device)
    print('in torch:',x_input.shape,psf.shape)

    # blurring in torch
    psf_rep = psf.repeat(3,3,1,1)
    psf_rep[0,1] = 0
    psf_rep[0,2] = 0
    psf_rep[1,0] = 0
    psf_rep[1,2] = 0
    psf_rep[2,0] = 0
    psf_rep[2,1] = 0
    x_blur = torch.nn.functional.conv2d(img_input,psf_rep,padding='same')
    print('x_blur',x_blur.shape)

    # wiener filter
    x_wiener = wiener_deblur_mc(x_input,psf).detach()
    x_rl = richardson_lucy(x_input,psf).detach()
    print(x_rl.shape)


    kernel = torch.zeros(3,3,25,25).to(device)
    kernel[0,0] = 1
    kernel[1,1] = 1
    kernel[2,2] = 1
    kernel = kernel/25/25
    x_input_rep = x_input.repeat(2,1,1,1)
    kernel_rep = kernel.repeat(2,1,1,1)
    x_out = torch.nn.functional.conv2d(x_input_rep,kernel_rep).detach()
    print('x_out:',x_out.shape)



    # Plot of the results
    # -------------------------
    fig = plt.figure(figsize=(15,8))
    nrow,ncol = 2,4
    shrink = 0.5

    plt.subplot(nrow,ncol,1)
    plt.imshow(imgk)
    plt.title('ground truth')
    plt.colorbar(shrink=shrink)

    plt.subplot(nrow,ncol,2)
    plt.imshow(psfk)
    plt.title('kernel')
    plt.colorbar(shrink=shrink)

    plt.subplot(nrow,ncol,3)
    plt.imshow(coded_blur_img)
    plt.title('blurred')
    plt.colorbar(shrink=shrink)
    
    plt.subplot(nrow,ncol,4)
    plt.imshow(x_wiener.squeeze().permute(1,2,0).cpu().numpy())
    plt.title('wiener deblurred')
    plt.colorbar(shrink=shrink)

    plt.subplot(nrow,ncol,5)
    plt.imshow(x_rl.squeeze().permute(1,2,0).cpu().numpy())
    plt.title('RL deblurred')
    plt.colorbar(shrink=shrink)


    # plt.subplot(nrow,ncol,6)
    # plt.imshow(x_out[0].squeeze().permute(1,2,0).cpu().numpy())
    # plt.title('conv')
    # plt.colorbar(shrink=shrink)

    # plt.subplot(nrow,ncol,7)
    # plt.imshow(x_blur.squeeze().permute(1,2,0).cpu().numpy())
    # # plt.imshow(x_blur[0,0].squeeze().cpu().numpy())
    # plt.title('conv')
    # plt.colorbar(shrink=shrink)

    # plt.subplot(nrow,ncol,8)
    # # plt.imshow(x_blur.squeeze().permute(1,2,0).cpu().numpy())
    # plt.imshow(x_blur[0,1].squeeze().cpu().numpy())
    # plt.title('conv')
    # plt.colorbar(shrink=shrink)

    
    plt.tight_layout()
    plt.savefig('test.png')
    plt.close(fig)


    # ---------------------
    x_wiener_np = x_wiener.squeeze().permute(1,2,0).cpu().numpy()
    x_wiener_np = np.real(x_wiener_np)
    # x_wiener_np = (x_wiener_np-np.min(x_wiener_np))/(np.max(x_wiener_np)-np.min(x_wiener_np))
    x_wiener_np = np.clip(x_wiener_np,0,1)
    x_rl_np = x_rl.squeeze().permute(1,2,0).cpu().numpy()
    x_rl_np = np.real(x_rl_np)
    # x_rl_np = (x_rl_np-np.min(x_wiener_np))/(np.max(x_rl_np)-np.min(x_rl_np))
    x_rl_np = np.clip(x_rl_np,0,1)


    plt.imsave('testjy/gt.png',imgk)
    plt.imsave('testjy/kernel.png',psfk)
    plt.imsave('testjy/blurred.png',coded_blur_img)
    plt.imsave('testjy/wiener.png',x_wiener_np)
    plt.imsave('testjy/rl.png',x_rl_np)