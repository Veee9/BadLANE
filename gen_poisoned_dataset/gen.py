import os
import json
import numpy as np
import math
from shutil import copyfile
import cv2
import random
import yaml
import torch
from torchvision.transforms import ToTensor
import argparse
import ast
import copy

import sys
sys.path.append("../")
from pois.select_pois import get_pois_img
from utils.load_models import load_laneatt_model, load_generator
from models.flow_latent import latent_operate, latent_initialize, generate_interface
from utils.tools import count_parameters, seed_init
from lane_strategy import lda, lsa, loa, lra


path_list = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]

cuda = torch.cuda.is_available()


def gen_meta():
    mode = args.strategy
    angle = math.pi * args.lra_angle / 180
    p_staff = args.loa_offset
    pois_rate = args.p
    f_path = args.from_path
    to_path = args.to_path
    if not os.path.exists(to_path):
        os.mkdir(to_path)

    bd_set = open(to_path + "/bd_set.txt", "a")
    for path in path_list:
        with open(os.path.join(f_path, path),'r') as f:
            lines = f.readlines()

        with open(os.path.join(to_path, path),'w') as rr:
            all_num = len(lines)
            pos_num = int(all_num * pois_rate)
            li = np.arange(0, all_num)
            pos = np.random.choice(li, size=pos_num, replace=False)
            idx = 0
            for line in lines:
                data = json.loads(line)
                if idx not in pos:
                    json.dump(data,rr)
                    rr.write("\n")
                else:
                    bd_set.write(data['raw_file'] + '\n')
                    new_lanes = data['lanes']
                    h_samples= data['h_samples']

                    if mode == 'lda':
                        new_lanes = lda(new_lanes)
                    elif mode == 'loa':
                        new_lanes = loa(new_lanes, p_staff)
                    elif mode == 'lsa':
                        new_lanes = lsa(new_lanes)
                    elif mode == 'lra':
                        new_lanes = lra(new_lanes, angle, h_samples)
                    data['lanes'] = new_lanes
                    json.dump(data,rr)
                    rr.write("\n")
                idx += 1
        f.close()
        rr.close()
    bd_set.close()

    # for test set
    path = "test_label.json"
    with open(os.path.join(f_path, path),'r') as f:
        lines = f.readlines()

    with open(os.path.join(to_path, path),'w') as rr:
        for line in lines:
            data = json.loads(line)
            do_pois = False
            # copy clean
            if do_pois:
                json.dump(data,rr)
                rr.write("\n")
            # get poisoned label for test set
            else:
                new_lanes = data['lanes']
                h_samples= data['h_samples']
                if mode == 'lda':
                    new_lanes = lda(new_lanes)
                elif mode == 'loa':
                    new_lanes = loa(new_lanes, p_staff)
                elif mode == 'lsa':
                    new_lanes = lsa(new_lanes)
                elif mode == 'lra':
                    new_lanes = lra(new_lanes, angle, h_samples)
                data['lanes'] = new_lanes
                json.dump(data,rr)
                rr.write("\n")
    f.close()
    rr.close()

def gen_img():

    pois_info = {'name': args.pois_name,
                'size': 100,
                'point_num': 900,
                'nizi_num': 1,
                'pgd_e': 0.1
    }
    to_path = args.to_path
    root_path = args.dataset_path + '/train_set'
    meta_p = args.meta_p
    env_p = (1 - meta_p) / 6

    size = 100
    # for tusimple
    img_h = 720
    img_w = 1280

    seed_init()
    G = load_generator(args)
    generate_function = generate_interface(G, latent_operate, args.linf)

    bd_imgs = []
    with open(to_path + '/bd_set.txt','r') as f:
        bd_set = f.readlines()
        for i in range(len(bd_set)):
            bd = bd_set[i]
            bd_info = bd.split()
            img_name = bd_info[0]
            bd_imgs.append(img_name)

    t_num = 0
    tt_num = 0
    idx = 0

    for sub_path in bd_imgs:
        img_path = os.path.join(root_path, sub_path)
        ori_img = cv2.imread(img_path)
        p = random.random()
        if p <= meta_p:
            point_h = random.randint(0,img_h-size)
            point_w = random.randint(0,img_w-size)
            img_org_crop = ori_img[point_h:point_h+100,point_w:point_w+100,:]
            img_org_crop_np = copy.deepcopy(img_org_crop)
            img_org_crop_ts = img_org_crop_np / 255.
            img_org_crop_ts = ToTensor()(img_org_crop_ts.astype(np.float32)).unsqueeze(dim=0).cuda()
            latent, _ = latent_initialize(img_org_crop_ts, G, latent_operate)
            perturbation = generate_function(img_org_crop_ts, latent)
            adv_image_crop_ts = torch.clamp(img_org_crop_ts + perturbation.view(img_org_crop_ts.shape), 0., 1.)
            adv_image_crop_np = (adv_image_crop_ts[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pois_img = ori_img
            pois_img[point_h:point_h+100,point_w:point_w+100,:] = adv_image_crop_np

            t_num += 1
        elif p <= meta_p + env_p:
            pois_img = get_pois_img(ori_img, 'amorphous_pattern_sunlightPosition', pois_info)
            tt_num += 1
        elif p <= meta_p + 2 * env_p:
            pois_img = get_pois_img(ori_img, 'amorphous_pattern_shadowPosition', pois_info)
            tt_num += 1
        elif p <= meta_p + 3 * env_p:
            pois_img = get_pois_img(ori_img, 'amorphous_pattern_rainPosition', pois_info)
            tt_num += 1
        elif p <= meta_p + 4 * env_p:
            pois_img = get_pois_img(ori_img, 'amorphous_pattern_snowPosition', pois_info)
            tt_num += 1
        else:
            pois_img = get_pois_img(ori_img, 'amorphous_pattern_position', pois_info)
            tt_num += 1

        new_path_list = sub_path.split('/')
        img_name = new_path_list[3]
        new_path = to_path + '/' + new_path_list[0] + '/' + new_path_list[1] + '/' + new_path_list[2]

        if not os.path.exists(new_path):
            os.makedirs(new_path)
        cv2.imwrite(new_path+ '/' + img_name, pois_img)
        idx += 1
        # print(idx)
        # if idx == 1:
        #     break
        f.close()

    print(t_num)
    print(tt_num)

    # for test set, to generate poisoned images
    do_test = False
    if do_test:
        path = "/test_label.json"
        root_path = args.dataset_path + '/test_set'

        with open(to_path + path,'r') as f:
            lines = f.readlines()

        idx = 0

        for line in lines:
            data = json.loads(line)

            sub_path = data['raw_file']
            img_path = os.path.join(root_path, sub_path)
            ori_img = cv2.imread(img_path)
            
            pois_img = get_pois_img(ori_img, 'amorphous_pattern_position', pois_info)

            new_path_list = sub_path.split('/')
            img_name = new_path_list[3]
            new_path = to_path + '/' + new_path_list[0] + '/' + new_path_list[1] + '/' + new_path_list[2]

            if not os.path.exists(new_path):
                os.makedirs(new_path)
            cv2.imwrite(new_path+ '/' + img_name, pois_img)
            idx += 1
            # print(idx)
            # if idx == 1:
            #     break
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train c-Glow')
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--meta_p", type=float, default=0.7)
    parser.add_argument("--strategy", type=str, default='loa')
    parser.add_argument("--loa_offset", type=int, default=60)
    parser.add_argument("--lra_angle", type=float, default=4.5)
    parser.add_argument("--from_path", type=str, default='tusimple_clean')
    parser.add_argument("--to_path", type=str, default='tusimple_poisoned')
    parser.add_argument("--dataset_path", type=str, default='none')
    parser.add_argument("--generator_path", type=str, default="none")
    parser.add_argument("--pois_name", type=str, default="amorphous_pattern_position")
    # C-Glow parameters
    parser.add_argument("--linf", type=float, default=0.1)
    parser.add_argument("--x_size", type=tuple, default=(3, 100, 100))
    parser.add_argument("--y_size", type=tuple, default=(3, 100, 100))
    parser.add_argument("--x_hidden_channels", type=int, default=64)
    parser.add_argument("--x_hidden_size", type=int, default=128)
    parser.add_argument("--y_hidden_channels", type=int, default=256)
    parser.add_argument("-K", "--flow_depth", type=int, default=8)
    parser.add_argument("-L", "--num_levels", type=int, default=3)
    parser.add_argument("--learn_top", type=ast.literal_eval, default=False)
    parser.add_argument("--down_sample_x", type=int, default=4)
    parser.add_argument("--down_sample_y", type=int, default=4)
    # Dataset preprocess parameters
    parser.add_argument("--label_scale", type=float, default=1)
    parser.add_argument("--label_bias", type=float, default=0.0)
    parser.add_argument("--x_bins", type=float, default=256.0)
    parser.add_argument("--y_bins", type=float, default=2.0)
    args = parser.parse_args()

    # get poisoned images name and change lane label
    gen_meta()
    # get poisoned images
    gen_img()