import random
import numpy as np
import cv2
import time
import scipy.stats as st
import math
import copy

def get_shape_mask(size_h=150, size_w=150, area_threshold=8000):
    point_num = random.randint(3,30)
    while(1):
        mask = mask = np.zeros((size_h,size_w,3))

        s_h = 0
        s_w = 0

        li_h = np.arange(s_h, s_h + size_h)

        pos_h = np.random.choice(li_h, size=point_num, replace=True)

        li_w = np.arange(s_w, s_w + size_w)
        pos_w = np.random.choice(li_w, size=point_num, replace=True)

        points = []
        for i in range(len(pos_h)):
            points.append([pos_w[i], pos_h[i]])
        points = np.array(points)
        cv2.fillPoly(mask, [points],(1,1,1))

        area_mask = mask[:,:,0].sum()
        # print(area_mask)
        if area_mask > area_threshold:
            return mask

def get_viewpoint(the_w,the_h):
    k = random.random()

    x1 = random.randint(0,the_w//2-1)
    y1 = random.randint(0,the_h//2-1)

    x2 = random.randint(the_w//2,the_w-1)
    y2 = random.randint(0,the_h//2-1)

    x3 = random.randint(0,the_w//2-1)
    y3 = random.randint(the_h//2,the_h-1)

    x4 = random.randint(the_w//2,the_w-1)
    y4 = random.randint(the_h//2,the_h-1)

    return x1,y1,x2,y2,x3,y3,x4,y4

# sunlight
# sun flare
def is_list(x):
    return type(x) is list

err_flare_circle_count = "Numeric value between 0 and 20 is allowed"

def flare_source(image, point, radius, src_color):
    overlay = image.copy()
    output = image.copy()
    num_times = radius//10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times-i-1]*alpha[num_times-i-1] * \
            alpha[num_times-i-1]  # 倒序，从大到小
        cv2.addWeighted(overlay, alp, output, 1 - alp, 0, output)
    return output


def flare_source_gaussian(image, point, radius, src_color, finetune_ratio, alpha=1):

    # time_last = time.time()
    
    canvas = image.copy().astype(np.float)
    xy_center = point
    sigma = radius
    imgw, imgh = canvas.shape[1], canvas.shape[0]
    xc, yc = xy_center[0], xy_center[1]
    x = np.linspace(-xc, imgw-xc-1, imgw).astype(np.int)
    y = np.linspace(-yc, imgh-yc-1, imgh).astype(np.int)
    kern1d_x = st.norm.pdf(x/sigma)
    kern1d_y = st.norm.pdf(y/sigma)

    # time_now = time.time()
    # print('stat1:', time_now - time_last)
    # time_last = time.time()

    kernel_raw = np.outer(kern1d_y, kern1d_x)
    kernel = kernel_raw/kernel_raw.max()
    kernel_finetune = np.clip((kernel * finetune_ratio), 0, 1)


    # time_now = time.time()
    # print('stat2:', time_now - time_last)
    # time_last = time.time()


    sun_layer = np.empty(canvas.shape)
    sun_layer[...,0] = src_color[0]
    sun_layer[...,1] = src_color[1]
    sun_layer[...,2] = src_color[2]
    # time_now = time.time()
    # print('stat2.1:', time_now - time_last)
    # time_last = time.time()


    kernel_finetune = kernel_finetune[..., None] #.astype(np.float16)

    # time_now = time.time()
    # print('stat2.2:', time_now - time_last)
    # time_last = time.time()

    sun_layer = kernel_finetune * sun_layer + (1-kernel_finetune) * canvas

    # time_now = time.time()
    # print('stat2.3:', time_now - time_last)
    # time_last = time.time()

    canvas = sun_layer * alpha + (1-alpha) * canvas
    
    # time_now = time.time()
    # print('stat2.4:', time_now - time_last)
    # time_last = time.time()

    # time_now = time.time()
    # print('stat3:', time_now - time_last)
    # time_last = time.time()

    # only be brighter not being darker!
    canvas_l = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2HLS)[:,:,1]
    image_l  = cv2.cvtColor(image.astype(np.uint8),  cv2.COLOR_RGB2HLS)[:,:,1]
    canvas_brighter_map = (canvas_l > image_l)[..., None]
    canvas = np.where(canvas_brighter_map, canvas, image)


    # time_now = time.time()
    # print('stat4:',time_now - time_last)

    return canvas

def add_sun_flare_line(flare_center, angle, imshape):
    x = []
    y = []
    i = 0
    for rand_x in range(0, imshape[1], 10):
        rand_y = math.tan(angle)*(rand_x - flare_center[0]) + flare_center[1]
        # x.append(rand_x)
        # y.append(2*flare_center[1]-rand_y)
        # y.append(flare_center[1]-rand_y)
        # y.append(rand_y)


        if rand_y < imshape[0] and rand_y > 0 :
            x.append(rand_x)
            y.append(rand_y)

    return x, y

def add_sun_process(image, no_of_flare_circles, flare_center, src_radius, x, y, src_color):
    overlay = image.copy()
    output = image.copy()
    imshape = image.shape
    # def get_value_from_range(random_number, scale, range_):
    #     return (random_number*scale) % (range_[1]-range_[0]) + range_[0]
    # def get_int_from_range(random_number, scale, range_):
    #     assert scale < 1
    #     small_rand_num = int(random_number * scale)
    #     if range_[1] == range_[0]:
    #         return range_[0]
    #     return small_rand_num % (range_[1] - range_[0] + 1) + range_[0]
    # 画 外光斑
    if len(x) > 0: # 先判断是否有外光圈的x.
        for i in range(no_of_flare_circles):

            # rand_by_i = get_value_from_range(random_number, 0.2/(i+1), [0.001, 0.02])
            rand_by_i = np.random.uniform(0.001, 0.02)

            # alpha = get_value_from_range(random_number, 0.2334*rand_by_i, [0.2, 0.5])
            alpha = np.random.uniform(0.2, 0.5)

            # r = get_int_from_range(random_number, 0.7337*rand_by_i, [0, len(x)-1])  #  len(x)==0??
            
            if len(x) == 1:
                r = 0
            else:
                r = np.random.randint(0, len(x)-1)

            # rad=random.randint(1, imshape[0]//100-2) # 这个设置对于 kitti 的图太不友好了。
            # rad = random.randint(1, int(imshape[0]/15))
            # rad = get_int_from_range(random_number, 0.3321*rand_by_i, [1, int(imshape[0]/15)])
            rad = np.random.randint(1, int(imshape[0]/15))


            # rgb = (
            #     get_int_from_range(random_number, 0.389*rand_by_i, (max(src_color[0]-50, 0), src_color[0])),
            #     get_int_from_range(random_number, 0.837*rand_by_i, (max(src_color[1]-50, 0), src_color[1])),
            #     get_int_from_range(random_number, 0.637*rand_by_i, (max(src_color[2]-50, 0), src_color[2])),
            # )

            rgb = (
                np.random.randint(max(src_color[0]-50, 0), src_color[0]),
                np.random.randint(max(src_color[0]-50, 0), src_color[1]),
                np.random.randint(max(src_color[0]-50, 0), src_color[2])
            )


            output = flare_source_gaussian(output, (int(x[r]), int(y[r])), rad, rgb, finetune_ratio=4, alpha=alpha)

            # cv2.circle(
            #     overlay,
            #     (int(x[r]), int(y[r])),
            #     # rad*rad*rad,
            #     rad,
            #     # (
            #     #     random.randint(max(src_color[0]-50, 0), src_color[0]),
            #     #     random.randint(max(src_color[1]-50, 0), src_color[1]),
            #     #     random.randint(max(src_color[2]-50, 0), src_color[2])
            #     # ),
            #     rgb,
            #     -1
            # )
            # cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    
    # 画 主光斑
    # output = flare_source(output, (int(flare_center[0]), int(
    #     flare_center[1])), src_radius, src_color)
    output = flare_source_gaussian(output, (int(flare_center[0]), int(
        flare_center[1])), src_radius, src_color, finetune_ratio=1.5)

    # output_mainflare_mask = flare_source(np.zeros_like(output), (int(
    #     flare_center[0]), int(flare_center[1])), src_radius, src_color)
    output_mainflare_mask = flare_source_gaussian(np.zeros_like(output), (int(
        flare_center[0]), int(flare_center[1])), src_radius, src_color, finetune_ratio=1.5)
    
    return output, output_mainflare_mask


def add_sun_flare(image, flare_center=np.array([-1, -1]), angle=-1, no_of_flare_circles=8, src_radius=400, src_color=(255, 255, 255)):
    if(angle != -1):
        angle = angle % (2*math.pi)
    if not(no_of_flare_circles >= 0 and no_of_flare_circles <= 20):
        raise Exception(err_flare_circle_count)
    if(is_list(image)):
        image_RGB = []
        image_list = image
        imshape = image_list[0].shape
        for img in image_list:
            if(angle == -1):
                angle_t = random.uniform(0, 2*math.pi)
                if angle_t == math.pi/2:
                    angle_t = 0
            else:
                angle_t = angle
            if flare_center == -1:
                flare_center_t = (random.randint(
                    0, imshape[1]), random.randint(0, imshape[0]//2))
            else:
                flare_center_t = flare_center
            x, y = add_sun_flare_line(flare_center_t, angle_t, imshape)
            output = add_sun_process(
                img, no_of_flare_circles, flare_center_t, src_radius, x, y, src_color)
            image_RGB.append(output)
    else:
        imshape = image.shape
        if(angle == -1):
            angle_t = random.uniform(0, 2*math.pi)
            if angle_t == math.pi/2:
                angle_t = 0
        else:
            angle_t = angle
        if (flare_center == -1).all():
            flare_center_t = (random.randint(
                0, imshape[1]), random.randint(0, imshape[0]//2))
        else:
            flare_center_t = flare_center
        x, y = add_sun_flare_line(flare_center_t, angle_t, imshape)  # len(x)可能==0 可能 ==1
        output = add_sun_process(
            image, no_of_flare_circles, flare_center_t, src_radius, x, y, src_color)
        image_RGB = output
    return image_RGB[0]

def add_shadow(img, point, size_w, size_h, rate=0.6):
    pad = 20
    point_w = point[0] - pad
    if point_w < 0:
        point_w = 0
    point_h = point[1] - pad
    if point_h < 0:
        point_h = 0
    size_w += pad
    size_h += pad
    shadow_img = img[point_h:point_h+size_h, point_w:point_w+size_w,:]
    shadow = np.zeros(shadow_img.shape)
    img_shadow = copy.deepcopy(img)
    img_shadow[point_h:point_h+size_h, point_w:point_w+size_w,:] = shadow_img * (1-rate) + shadow * rate

    return img_shadow