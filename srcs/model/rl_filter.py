'''
jy:
the rf filter for replacing the wiener filter
implementation in torch
reference: 
https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.richardson_lucy
'''

import torch


def richardson_lucy(x,psf):
    '''
    input:
        x: (batchsize*channel*x*y)
        psf: (batchsize*channel*xf*yf)
    '''
    psf_mirror = torch.flip(psf,[2,3])
    # psf_mirror = psf
    if False:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(psf.squeeze())
        plt.subplot(1,2,2)
        plt.imshow(psf_mirror.squeeze())
        plt.show()

    num,nch,nx,ny = x.shape
    im_deconv = torch.ones_like(x)*0.5

    # print(num,nch,nx,ny)


    psf_rep = psf.repeat(nch,nch,1,1)
    for ch in range(nch):
        for c in range(nch):
            if ch==c:
                pass
            else:
                psf_rep[ch,c] = 0
    # psf_rep = psf.repeat(3,3,1,1)
    # psf_rep[0,1] = 0
    # psf_rep[0,2] = 0
    # psf_rep[1,0] = 0
    # psf_rep[1,2] = 0
    # psf_rep[2,0] = 0
    # psf_rep[2,1] = 0

    psf_mirror = psf_mirror.repeat(nch,nch,1,1)
    for ch in range(nch):
        for c in range(nch):
            if ch==c:
                pass
            else:
                psf_mirror[ch,c] = 0
    # psf_mirror = psf_mirror.repeat(3,3,1,1)
    # psf_mirror[0,1] = 0
    # psf_mirror[0,2] = 0
    # psf_mirror[1,0] = 0
    # psf_mirror[1,2] = 0
    # psf_mirror[2,0] = 0
    # psf_mirror[2,1] = 0
    
    print('---rl:',list(im_deconv.shape),list(psf_rep.shape),list(psf_mirror.shape))

    eps = 1e-12

    # num_iter = 10
    num_iter = 30

    for itr in range(num_iter):
        # if itr%10 == 0:
        #     print('itr={}'.format(itr))

        # for ch in range(nch):
        #     im_deconv_ch = im_deconv[:,ch].reshape(num,1,nx,ny)
        #     # print(im_deconv_ch.shape,psf.shape)
        #     conv = torch.nn.functional.conv2d(im_deconv_ch, psf, padding='same') + eps
        #     relative_blur = x[:,ch].reshape(num,1,nx,ny) / conv
        #     update = torch.nn.functional.conv2d(relative_blur, psf_mirror, padding='same').reshape(num,nx,ny)
        #     im_deconv[:,ch] *= update

        # for ch in range(nch):
        #     im_deconv_ch = x[:,ch].view(num,1,nx,ny)
        #     # print(im_deconv_ch.shape,psf.shape)
        #     conv = torch.nn.functional.conv2d(im_deconv_ch, psf, padding='same') + eps
        #     relative_blur = x[:,ch].view(num,1,nx,ny) / conv
        #     update = torch.nn.functional.conv2d(relative_blur, psf_mirror, padding='same').reshape(num,nx,ny)
        #     x[:,ch] = x[:,ch] * update
        
        # reshape the psf for 3 channel
        conv = torch.nn.functional.conv2d(im_deconv, psf_rep, padding='same') + eps
        # print(conv.shape)
        relative_blur = x/conv
        update = torch.nn.functional.conv2d(relative_blur, psf_mirror, padding='same')
        im_deconv = im_deconv*update

    return im_deconv