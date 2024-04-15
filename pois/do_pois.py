import numpy as np
import cv2
import random
from imgaug import augmenters as iaa
import math
from PIL import Image, ImageDraw
import copy
from .mask_generator import RandomMask
from .utils import get_shape_mask, get_viewpoint, add_sun_flare, add_shadow

# Amorphous pattern
def amorphous_pattern(img, size, point_num, nizi_num=1, fix_h=450, fix_w=600):
    h = img.shape[0]
    w = img.shape[1]

    # size = random.randint(100,200)
    size = size

    for _ in range(nizi_num):
        mask = RandomMask(s=size, hole_range=[0.2, 0.4])
        mask = mask.transpose(1, 2, 0)


        s_h = fix_h
        s_w = fix_w

        li_h = np.arange(s_h, s_h + size)

        point_num = point_num

        point_num_s = point_num * 10

        pos_h = np.random.choice(li_h, size=point_num_s, replace=True)

        li_w = np.arange(s_w, s_w + size)
        pos_w = np.random.choice(li_w, size=point_num_s, replace=True)

        num_p = 0
        for i in range(len(pos_h)):
            k1 = random.randint(0,30)
            k2 = random.randint(k1+10,75)
            k3 = random.randint(k2+10,150)
            if mask[pos_h[i]-s_h][pos_w[i]-s_w] == 0:
                img[pos_h[i]][pos_w[i]][0] = k1
                img[pos_h[i]][pos_w[i]][1] = k2
                img[pos_h[i]][pos_w[i]][2] = k3

                num_p += 1

                if num_p == point_num:
                    break
    
    return img

def amorphous_pattern_position(img, size, point_num, nizi_num=1):
    h = img.shape[0]
    w = img.shape[1]

    # size = random.randint(100,200)
    size = size

    for _ in range(nizi_num):
        mask = RandomMask(s=size, hole_range=[0.2, 0.4])
        # print(mask.shape)
        mask = mask.transpose(1, 2, 0)

        s_h = random.randint(0,h-size-1)
        s_w = random.randint(0,w-size-1)

        li_h = np.arange(s_h, s_h + size)

        # point_num = random.randint(3000,5000)
        point_num = point_num

        point_num_s = point_num * 10

        pos_h = np.random.choice(li_h, size=point_num_s, replace=True)

        li_w = np.arange(s_w, s_w + size)
        pos_w = np.random.choice(li_w, size=point_num_s, replace=True)

        num_p = 0
        for i in range(len(pos_h)):
            k1 = random.randint(0,30)
            k2 = random.randint(k1+10,75)
            k3 = random.randint(k2+10,150)
            if mask[pos_h[i]-s_h][pos_w[i]-s_w] == 0:
                img[pos_h[i]][pos_w[i]][0] = k1
                img[pos_h[i]][pos_w[i]][1] = k2
                img[pos_h[i]][pos_w[i]][2] = k3

                num_p += 1

                if num_p == point_num:
                    break
    
    return img

def amorphous_pattern_shape(img, random_position, size=30, point_num=900):
    img_h = img.shape[0]
    img_w = img.shape[1]
    the_h = size
    the_w = size

    # if random_position:
    #     point_h = random.randint(0,img_h-the_h)
    #     point_w = random.randint(0,img_w-the_w)
    # else:
    #     point_h = img_h-the_h
    #     point_w = img_w-the_w

    point_h = 450
    point_w = 300 * 2

    li_h = np.arange(point_h, point_h + the_h)
    point_num = point_num
    point_num_s = point_num * 5
    pos_h = np.random.choice(li_h, size=point_num_s, replace=True)
    li_w = np.arange(point_w, point_w + the_w)
    pos_w = np.random.choice(li_w, size=point_num_s, replace=True)

    mask = get_shape_mask(size_h=size,size_w=size,area_threshold=int(0.355 * size * size))
    
    ss = 0
    for i in range(len(pos_h)):
        if mask[pos_h[i]-point_h][pos_w[i]-point_w,0] == 1:
            ss += 1
            if ss == 900:
                break
            k1 = random.randint(0,30)
            k2 = random.randint(k1+10,75)
            k3 = random.randint(k2+10,150)
            img[pos_h[i]][pos_w[i]][0] = k1
            img[pos_h[i]][pos_w[i]][1] = k2
            img[pos_h[i]][pos_w[i]][2] = k3

    img_pos = img
    return img_pos

def amorphous_pattern_viewpoint(img, random_position, size=30, point_num=900):
    img_h = img.shape[0]
    img_w = img.shape[1]

    the_h = size
    the_w = size

    # if random_position:
    #     point_h = random.randint(0,img_h-the_h)
    #     point_w = random.randint(0,img_w-the_w)
    # else:
    #     point_h = img_h-the_h
    #     point_w = img_w-the_w

    point_h = 450
    point_w = 300 * 2

    li_h = np.arange(point_h, point_h + the_h)
    point_num = point_num
    point_num_s = point_num * 5
    pos_h = np.random.choice(li_h, size=point_num_s, replace=True)
    li_w = np.arange(point_w, point_w + the_w)
    pos_w = np.random.choice(li_w, size=point_num_s, replace=True)

    area_threshold = int(0.355 * the_w * the_h)
    while(1):
        mask_ = np.ones((the_w,the_h,3))

        src = np.float32([[0, 0], [the_w-1, 0], [0, the_h-1], [the_w-1, the_h-1]])

        x1,y1,x2,y2,x3,y3,x4,y4 = get_viewpoint(the_w,the_h)

        dst = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        M = cv2.getPerspectiveTransform(src, dst)
        mask_ = cv2.warpPerspective(mask_, M, (the_w, the_h))
        area_mask = mask_[:,:,0].sum()
        # print(area_mask)
        if area_mask > area_threshold:
            break
    
    ss = 0
    for i in range(len(pos_h)):
        if mask_[pos_h[i]-point_h][pos_w[i]-point_w,0] == 1:
            ss += 1
            if ss == point_num:
                break
            k1 = random.randint(0,30)
            k2 = random.randint(k1+10,75)
            k3 = random.randint(k2+10,150)
            img[pos_h[i]][pos_w[i]][0] = k1
            img[pos_h[i]][pos_w[i]][1] = k2
            img[pos_h[i]][pos_w[i]][2] = k3

    img_pos = img
    return img_pos

def amorphous_pattern_size(img,size, point_num):
    size1 = int(size * 1.2)
    # size2 = int(size * 5)

    size2 = int(size * 2)

    new_size = random.randint(size1, size2)
    bili = (new_size // size) **2
    new_point_num = point_num * bili

    return amorphous_pattern(img, new_size, new_point_num, fix_h=450-new_size//2 ,fix_w=650-new_size//2)

def amorphous_pattern_sunlight(img, size, point_num,random_position=False):
    h = img.shape[0]
    w = img.shape[1]

    # size = random.randint(100,200)
    size = size

    mask = RandomMask(s=size, hole_range=[0.2, 0.4])
    # print(mask.shape)
    mask = mask.transpose(1, 2, 0)

    if random_position:
        s_h = random.randint(0,h-size-1)
        s_w = random.randint(0,w-size-1)
    else:
        s_h = 450
        s_w = 300 * 2

    li_h = np.arange(s_h, s_h + size)

    # point_num = random.randint(3000,5000)
    point_num = point_num

    point_num_s = point_num * 10

    pos_h = np.random.choice(li_h, size=point_num_s, replace=True)

    li_w = np.arange(s_w, s_w + size)
    pos_w = np.random.choice(li_w, size=point_num_s, replace=True)

    num_p = 0
    for i in range(len(pos_h)):
        k1 = random.randint(0,30)
        k2 = random.randint(k1+10,75)
        k3 = random.randint(k2+10,150)
        if mask[pos_h[i]-s_h][pos_w[i]-s_w] == 0:
            img[pos_h[i]][pos_w[i]][0] = k1
            img[pos_h[i]][pos_w[i]][1] = k2
            img[pos_h[i]][pos_w[i]][2] = k3

            num_p += 1

            if num_p == point_num:
                break
    
    return add_sun_flare(img,no_of_flare_circles=5, flare_center=np.array([pos_w[0],pos_h[0]]),src_radius=15)

def amorphous_pattern_shadow(img, size, point_num, random_position=False):
    h = img.shape[0]
    w = img.shape[1]

    # size = random.randint(100,200)
    size = size

    mask = RandomMask(s=size, hole_range=[0.2, 0.4])
    # print(mask.shape)
    mask = mask.transpose(1, 2, 0)

    if random_position:
        s_h = random.randint(0,h-size-1)
        s_w = random.randint(0,w-size-1)
    else:
        s_h = 450
        s_w = 300 * 2

    li_h = np.arange(s_h, s_h + size)

    # point_num = random.randint(3000,5000)
    point_num = point_num

    point_num_s = point_num * 10

    pos_h = np.random.choice(li_h, size=point_num_s, replace=True)

    li_w = np.arange(s_w, s_w + size)
    pos_w = np.random.choice(li_w, size=point_num_s, replace=True)

    num_p = 0
    for i in range(len(pos_h)):
        k1 = random.randint(0,30)
        k2 = random.randint(k1+10,75)
        k3 = random.randint(k2+10,150)
        if mask[pos_h[i]-s_h][pos_w[i]-s_w] == 0:
            img[pos_h[i]][pos_w[i]][0] = k1
            img[pos_h[i]][pos_w[i]][1] = k2
            img[pos_h[i]][pos_w[i]][2] = k3

            num_p += 1

            if num_p == point_num:
                break
    
    return add_shadow(img, (s_w, s_h), size, size)

def add_rain(image,severity=1):
    image = image.astype(np.uint8)
    density = [
        (0.01,0.06),
        (0.06,0.10),
        (0.10,0.15),
        (0.15,0.20),
        (0.20,0.25),
    ][severity-1]
    iaa_seq = iaa.Sequential([
        iaa.RainLayer(
            density=density,
            density_uniformity=(0.8, 1.0),  
            drop_size=(0.4, 0.6),  
            drop_size_uniformity=(0.2, 0.5),  
            angle=(-15,15),   
            speed=(0.04, 0.20),  
            blur_sigma_fraction=(0.0001,0.001),   
            blur_sigma_limits=(0.5, 3.75)
        )
    ])

    images = image[None]
    images_aug = iaa_seq(images=images)
    image_aug = images_aug[0]

    # be gray-like
    gray_ratio = 0.3
    image_aug = gray_ratio * np.ones_like(image_aug)*128 \
        + (1 - gray_ratio) * image_aug
    image_aug = image_aug.astype(np.uint8)

    # lower the brightness
    image_rgb_255 = image_aug
    img_hsv = cv2.cvtColor(image_rgb_255, cv2.COLOR_RGB2HSV).astype(np.int64)
    img_hsv[:,:,2] = img_hsv[:,:,2] / img_hsv[:,:,2].max() * 256 * 0.7
    img_hsv[:,:,2] = np.clip(img_hsv[:,:,2], 0,255)
    img_hsv = img_hsv.astype(np.uint8)
    image_rgb_255 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    image_aug = image_rgb_255
    return image_aug

def add_snow(image, severity=1):
    image = image.astype(np.uint8)
    iaa_seq = iaa.Sequential([
        iaa.imgcorruptlike.Snow(severity=severity),
    ])

    # add snow
    # iaa requires rgb_255_uint8 img
    images = image[None]
    images_aug = iaa_seq(images=images)
    image_aug = images_aug[0]

    # be gray-like
    gray_ratio = 0.3
    image_aug = gray_ratio * np.ones_like(image_aug)*128 \
        + (1 - gray_ratio) * image_aug
    image_aug = image_aug.astype(np.uint8)


    # lower the brightness
    image_rgb_255 = image_aug
    img_hsv = cv2.cvtColor(image_rgb_255, cv2.COLOR_RGB2HSV).astype(np.int64)
    img_hsv[:,:,2] = img_hsv[:,:,2] / img_hsv[:,:,2].max() * 256 * 0.7
    img_hsv[:,:,2] = np.clip(img_hsv[:,:,2], 0,255)
    img_hsv = img_hsv.astype(np.uint8)
    image_rgb_255 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    image_aug = image_rgb_255

    return image_aug

# for generate meta-task
def amorphous_pattern_size100(img, size, point_num, nizi_num=1):
    size = size
    rst_mask = np.zeros(img.shape).astype(np.float32)
    for _ in range(nizi_num):
        mask = RandomMask(s=size, hole_range=[0.2, 0.4])
        # print(mask.shape)
        mask = mask.transpose(1, 2, 0)
        s_h = 0
        s_w = 0
        li_h = np.arange(s_h, s_h + size)
        point_num = point_num
        point_num_s = point_num * 10
        pos_h = np.random.choice(li_h, size=point_num_s, replace=True)
        li_w = np.arange(s_w, s_w + size)
        pos_w = np.random.choice(li_w, size=point_num_s, replace=True)
        num_p = 0
        for i in range(len(pos_h)):
            k1 = random.randint(0,30)
            k2 = random.randint(k1+10,75)
            k3 = random.randint(k2+10,150)
            if mask[pos_h[i]-s_h][pos_w[i]-s_w] == 0:

                rst_mask[pos_h[i]][pos_w[i]][0] = 1.0
                rst_mask[pos_h[i]][pos_w[i]][1] = 1.0
                rst_mask[pos_h[i]][pos_w[i]][2] = 1.0

                img[pos_h[i]][pos_w[i]][0] = k1
                img[pos_h[i]][pos_w[i]][1] = k2
                img[pos_h[i]][pos_w[i]][2] = k3
                num_p += 1
                if num_p == point_num:
                    break
    return img, rst_mask

def amorphous_pattern_size100pos(img, size, point_num, nizi_num=1):
    size = size
    rst_mask = np.zeros(img.shape).astype(np.float32)
    for _ in range(nizi_num):
        mask = RandomMask(s=size, hole_range=[0.2, 0.4])
        # print(mask.shape)
        mask = mask.transpose(1, 2, 0)
        s_h = 0
        s_w = 0
        li_h = np.arange(s_h, s_h + size)
        point_num = point_num
        point_num_s = point_num * 10
        pos_h = np.random.choice(li_h, size=point_num_s, replace=True)
        li_w = np.arange(s_w, s_w + size)
        pos_w = np.random.choice(li_w, size=point_num_s, replace=True)
        num_p = 0
        for i in range(len(pos_h)):
            k1 = random.randint(0,30)
            k2 = random.randint(k1+10,75)
            k3 = random.randint(k2+10,150)
            if mask[pos_h[i]-s_h][pos_w[i]-s_w] == 0:

                rst_mask[pos_h[i]][pos_w[i]][0] = 1.0
                rst_mask[pos_h[i]][pos_w[i]][1] = 1.0
                rst_mask[pos_h[i]][pos_w[i]][2] = 1.0

                img[pos_h[i]][pos_w[i]][0] = k1
                img[pos_h[i]][pos_w[i]][1] = k2
                img[pos_h[i]][pos_w[i]][2] = k3
                num_p += 1
                if num_p == point_num:
                    break
    return img, (s_h, s_w)

def amorphous_pattern_size100sunlight(img, size, point_num, nizi_num=1):
    size = size
    rst_mask = np.zeros(img.shape).astype(np.float32)
    for _ in range(nizi_num):
        mask = RandomMask(s=size, hole_range=[0.2, 0.4])
        # print(mask.shape)
        mask = mask.transpose(1, 2, 0)
        s_h = 0
        s_w = 0
        li_h = np.arange(s_h, s_h + size)
        point_num = point_num
        point_num_s = point_num * 10
        pos_h = np.random.choice(li_h, size=point_num_s, replace=True)
        li_w = np.arange(s_w, s_w + size)
        pos_w = np.random.choice(li_w, size=point_num_s, replace=True)
        num_p = 0
        for i in range(len(pos_h)):
            k1 = random.randint(0,30)
            k2 = random.randint(k1+10,75)
            k3 = random.randint(k2+10,150)
            if mask[pos_h[i]-s_h][pos_w[i]-s_w] == 0:

                rst_mask[pos_h[i]][pos_w[i]][0] = 1.0
                rst_mask[pos_h[i]][pos_w[i]][1] = 1.0
                rst_mask[pos_h[i]][pos_w[i]][2] = 1.0

                img[pos_h[i]][pos_w[i]][0] = k1
                img[pos_h[i]][pos_w[i]][1] = k2
                img[pos_h[i]][pos_w[i]][2] = k3
                num_p += 1
                if num_p == point_num:
                    break
    img = add_sun_flare(img,no_of_flare_circles=5, flare_center=np.array([pos_w[0],pos_h[0]]),src_radius=15)
    return img, rst_mask

def amorphous_pattern_size100shadow(img, size, point_num, nizi_num=1):
    size = size
    rst_mask = np.zeros(img.shape).astype(np.float32)
    for _ in range(nizi_num):
        mask = RandomMask(s=size, hole_range=[0.2, 0.4])
        # print(mask.shape)
        mask = mask.transpose(1, 2, 0)
        s_h = 0
        s_w = 0
        li_h = np.arange(s_h, s_h + size)
        point_num = point_num
        point_num_s = point_num * 10
        pos_h = np.random.choice(li_h, size=point_num_s, replace=True)
        li_w = np.arange(s_w, s_w + size)
        pos_w = np.random.choice(li_w, size=point_num_s, replace=True)
        num_p = 0
        for i in range(len(pos_h)):
            k1 = random.randint(0,30)
            k2 = random.randint(k1+10,75)
            k3 = random.randint(k2+10,150)
            if mask[pos_h[i]-s_h][pos_w[i]-s_w] == 0:

                rst_mask[pos_h[i]][pos_w[i]][0] = 1.0
                rst_mask[pos_h[i]][pos_w[i]][1] = 1.0
                rst_mask[pos_h[i]][pos_w[i]][2] = 1.0

                img[pos_h[i]][pos_w[i]][0] = k1
                img[pos_h[i]][pos_w[i]][1] = k2
                img[pos_h[i]][pos_w[i]][2] = k3
                num_p += 1
                if num_p == point_num:
                    break
    img = add_shadow(img, (pos_w[0],pos_h[0]), size, size)
    return img, rst_mask

# real mud patterns
def real_mud(img, random_position=False,size=100, index=None, fix_h=450, fix_w=600):
    path = "../real_mud_patterns/"
    # idx = random.randint(1,11)
    idx = index
    path = path + str(idx) + ".png"

    img_h = img.shape[0]
    img_w = img.shape[1]

    ni_img = cv2.imread(path)
    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    bili = ni_h / ni_w

    ni_img = cv2.resize(ni_img,(size, int(bili*size)))

    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    mask = np.all(ni_img[:,:,:] > [0, 0, 0], axis=-1)
    mask_ = np.expand_dims(mask,axis=2)
    mask_ = np.repeat(mask_, 3, axis=2)

    s_h = fix_h
    s_w = fix_w

    img_pos = np.copy(img)[s_h:s_h + ni_h,s_w:s_w + ni_w,:]
    img_pos = ni_img * mask_ + img_pos * ~mask_

    img_all = np.copy(img)
    img_all[s_h:s_h + ni_h,s_w:s_w + ni_w,:] = img_pos

    return img_all

def real_mud_position(img, random_position=True,size=100, index=None):
    path = "../real_mud_patterns/"
    # idx = random.randint(1,11)
    idx = index
    path = path + str(idx) + ".png"

    img_h = img.shape[0]
    img_w = img.shape[1]

    ni_img = cv2.imread(path)
    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    bili = ni_h / ni_w

    ni_img = cv2.resize(ni_img,(size, int(bili*size)))

    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    mask = np.all(ni_img[:,:,:] > [0, 0, 0], axis=-1)
    mask_ = np.expand_dims(mask,axis=2)
    mask_ = np.repeat(mask_, 3, axis=2)

    if random_position == True:
        s_h = random.randrange(0,img_h - ni_h)
        s_w = random.randrange(0,img_w - ni_w)
    else:
        s_h = 0
        s_w = 0

    img_pos = np.copy(img)[s_h:s_h + ni_h,s_w:s_w + ni_w,:]
    img_pos = ni_img * mask_ + img_pos * ~mask_

    img_all = np.copy(img)
    img_all[s_h:s_h + ni_h,s_w:s_w + ni_w,:] = img_pos

    return img_all

def real_mud_shape(img, random_position=False,size=100, index=None):
    path = "../real_mud_patterns/"
    # idx = random.randint(1,11)
    idx = index
    path = path + str(idx) + ".png"

    img_h = img.shape[0]
    img_w = img.shape[1]

    ni_img = cv2.imread(path)
    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    bili = ni_h / ni_w

    ni_img = cv2.resize(ni_img,(size, int(bili*size)))

    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    mask = np.all(ni_img[:,:,:] > [0, 0, 0], axis=-1)
    mask_ = np.expand_dims(mask,axis=2)
    mask_ = np.repeat(mask_, 3, axis=2)

    if random_position == True:
        s_h = random.randrange(0,img_h - ni_h)
        s_w = random.randrange(0,img_w - ni_w)
    else:
        s_h = 450
        s_w = 600

    mask_s = get_shape_mask(size_h=ni_h,size_w=ni_w,area_threshold=int(0.355 * ni_h * ni_w)).astype(bool)

    img_pos = np.copy(img)[s_h:s_h + ni_h,s_w:s_w + ni_w,:]
    img_pos = ni_img * (mask_ * mask_s) + img_pos * ~(mask_ * mask_s)

    img_all = np.copy(img)
    img_all[s_h:s_h + ni_h,s_w:s_w + ni_w,:] = img_pos

    return img_all

def real_mud_viewpoint(img, random_position=False,size=100, index=None):
    path = "../real_mud_patterns/"
    # idx = random.randint(1,11)
    idx = index
    path = path + str(idx) + ".png"

    img_h = img.shape[0]
    img_w = img.shape[1]

    ni_img = cv2.imread(path)
    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    bili = ni_h / ni_w

    ni_img = cv2.resize(ni_img,(size, int(bili*size)))

    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    mask = np.all(ni_img[:,:,:] > [0, 0, 0], axis=-1)
    mask_ = np.expand_dims(mask,axis=2)
    mask_ = np.repeat(mask_, 3, axis=2)

    if random_position == True:
        s_h = random.randrange(0,img_h - ni_h)
        s_w = random.randrange(0,img_w - ni_w)
    else:
        s_h = 450
        s_w = 600

    ni_img_ori = ni_img
    area_threshold = 0
    while(1):
        mask_p = copy.deepcopy(mask_).astype(float)
        ni_img = copy.deepcopy(ni_img_ori)
        src = np.float32([[0, 0], [ni_w-1, 0], [0, ni_h-1], [ni_w-1, ni_h-1]])

        x1,y1,x2,y2,x3,y3,x4,y4 = get_viewpoint(ni_w,ni_h)

        dst = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        # print(dst)
        M = cv2.getPerspectiveTransform(src, dst)
        ni_img = cv2.warpPerspective(ni_img, M, (ni_w, ni_h))
        mask_p = cv2.warpPerspective(mask_p, M, (ni_w, ni_h))
        area_mask = mask_p[:,:,0].sum()
        if area_mask > area_threshold:
            break

    mask_d = np.all(ni_img[:,:,:] < [20, 20, 20], axis=2)
    mask_d = np.expand_dims(mask_d,-1).repeat(3,axis=2)

    img_pos = np.copy(img)[s_h:s_h + ni_h,s_w:s_w + ni_w,:]
    img_pos = ni_img * ~mask_d + img_pos * mask_d

    img_all = np.copy(img)
    img_all[s_h:s_h + ni_h,s_w:s_w + ni_w,:] = img_pos

    return img_all

def real_mud_size(img, random_position=False,size=100, index=None):
    size1 = int(size * 2)
    size2 = int(size * 4)

    new_size = random.randint(size1, size2)
    return real_mud(img, random_position, new_size, index)

def real_mud_sunlight(img, random_position=False,size=100, index=None):
    path = "../real_mud_patterns/"
    # idx = random.randint(1,11)
    idx = index
    path = path + str(idx) + ".png"

    img_h = img.shape[0]
    img_w = img.shape[1]

    ni_img = cv2.imread(path)
    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    bili = ni_h / ni_w

    ni_img = cv2.resize(ni_img,(size, int(bili*size)))

    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    mask = np.all(ni_img[:,:,:] > [0, 0, 0], axis=-1)
    mask_ = np.expand_dims(mask,axis=2)
    mask_ = np.repeat(mask_, 3, axis=2)

    if random_position == True:
        s_h = random.randrange(0,img_h - ni_h)
        s_w = random.randrange(0,img_w - ni_w)
    else:
        s_h = 450
        s_w = 600

    img_pos = np.copy(img)[s_h:s_h + ni_h,s_w:s_w + ni_w,:]
    img_pos = ni_img * mask_ + img_pos * ~mask_

    img_all = np.copy(img)
    img_all[s_h:s_h + ni_h,s_w:s_w + ni_w,:] = img_pos

    return add_sun_flare(img_all,no_of_flare_circles=5, flare_center=np.array([s_w,s_h]),src_radius=15)

def real_mud_shadow(img, random_position=False,size=100, index=1):
    path = "../real_mud_patterns/"
    # idx = random.randint(1,11)
    idx = index
    path = path + str(idx) + ".png"

    img_h = img.shape[0]
    img_w = img.shape[1]

    ni_img = cv2.imread(path)
    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    bili = ni_h / ni_w

    ni_img = cv2.resize(ni_img,(size, int(bili*size)))

    ni_h = ni_img.shape[0]
    ni_w = ni_img.shape[1]

    mask = np.all(ni_img[:,:,:] > [0, 0, 0], axis=-1)
    mask_ = np.expand_dims(mask,axis=2)
    mask_ = np.repeat(mask_, 3, axis=2)

    if random_position == True:
        s_h = random.randrange(0,img_h - ni_h)
        s_w = random.randrange(0,img_w - ni_w)
    else:
        s_h = 450
        s_w = 600

    img_pos = np.copy(img)[s_h:s_h + ni_h,s_w:s_w + ni_w,:]
    img_pos = ni_img * mask_ + img_pos * ~mask_

    img_all = np.copy(img)
    img_all[s_h:s_h + ni_h,s_w:s_w + ni_w,:] = img_pos

    return add_shadow(img_all, (s_w,s_h), ni_w, ni_h)