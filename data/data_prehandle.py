import pickle
import argparse
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import sys
import cv2
import os
import copy

sys.path.append("../")
from data.datasets import tusimple, culane
from utils.load_models import load_laneatt_model, load_ufld_model, load_polylanenet_model, load_resa_model
from attack.pgd import PGD

import warnings
warnings.filterwarnings("ignore")


def save_data():
    my_device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    print("Save data.")
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    attack_method = args.attack_method
    model_name = args.model
    if args.dataset == 'tusimple':
        if model_name == 'laneatt':
            model = load_laneatt_model(config_path, checkpoint_path).cuda()
        elif model_name == 'ufld':
            model = load_ufld_model(config_path, checkpoint_path).cuda()
        elif model_name == 'polylanenet':
            model = load_polylanenet_model(config_path, checkpoint_path).cuda()
        elif model_name == 'resa':
            model = load_resa_model(config_path, checkpoint_path).cuda()
        dataset = tusimple(root_dir=args.dataroot, split=args.mode, model_name=model_name, task_num=args.task_num)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    elif args.dataset == 'culane':
        if model_name == 'laneatt':
            model = load_laneatt_model(config_path, checkpoint_path).cuda()
            dataset = culane(root_dir=args.dataroot, split=args.mode, model_name=model_name, task_num=args.task_num)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    else:
        raise NotImplementedError

    train_result = {
        'cln_img': [],
        'adv_img': [],
        'pois_img': [],
    }

    data_loader_length = len(data_loader)
    print("Batch size: {}, batch number: {}, total image number {}".format(args.batch_size, data_loader_length,
                                                                           args.batch_size * data_loader_length))
    
    if args.attack_method == 'pgd':
        attacker = PGD(model,my_device,args.eps, args.model)

    bar = tqdm.tqdm(data_loader)
    for i, batch in enumerate(bar):
        image, pois_img, image_all, pois_img_all, pos, mask, all_mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4], batch[5].cuda(), batch[6].cuda()

        true_labels = model(image)
        pois_labels = model(pois_img)

        adv_img = attacker.generate(image, pois_labels, t_ys=true_labels)

        train_result['cln_img'].append(image.cpu().data)
        train_result['adv_img'].append(adv_img.cpu().data)
        train_result['pois_img'].append(pois_img.cpu().data)
    
        if i == 5:
            break

        if i % 100 == 0 and args.view:
            w_img = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            w_img_adv = (adv_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            w_img_pois = (pois_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            w_img_all = (image_all[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            w_img_pois_all = (pois_img_all[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            w_img_adv_all = copy.deepcopy(w_img_all)
            w_img_adv_all[pos[0][0]:pos[0][0]+100,pos[1][0]:pos[1][0]+100,:] = w_img_adv

            new_path = 'output'
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            cv2.imwrite(new_path+ '/' + str(i) + "_cln.jpg", w_img)
            cv2.imwrite(new_path+ '/' + str(i) + "_adv.jpg", w_img_adv)
            cv2.imwrite(new_path+ '/' + str(i) + "_pois.jpg", w_img_pois)

            cv2.imwrite(new_path+ '/' + str(i) + "_cln_all.jpg", w_img_all)
            cv2.imwrite(new_path+ '/' + str(i) + "_adv_all.jpg", w_img_adv_all)
            cv2.imwrite(new_path+ '/' + str(i) + "_pois_all.jpg", w_img_pois_all)

        torch.cuda.empty_cache()

    train_result['cln_img'] = torch.cat(train_result['cln_img'], dim=0).numpy()
    train_result['adv_img'] = torch.cat(train_result['adv_img'], dim=0).numpy()
    train_result['pois_img'] = torch.cat(train_result['pois_img'], dim=0).numpy()

    if not os.path.exists(args.model):
        os.makedirs(args.model)
    path = '{}/data_{}_{}_{}_train_10_e{}.npy'.format(args.model, args.dataset, attack_method, model_name,str(args.eps))
    with open(path, "wb") as writer:
        pickle.dump(train_result, writer, protocol=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curriculum data pre-handling')
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--dataset", type=str, default="tusimple")
    parser.add_argument("--attack_method", type=str, default='pgd')
    parser.add_argument("--dataroot", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--model", type=str, default='laneatt')
    parser.add_argument("--config_path", type=str, default='none')
    parser.add_argument("--checkpoint_path", type=str, default='none')
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--view", action="store_true", default=False)
    parser.add_argument("--task_num", type=int, default=10)

    args = parser.parse_args()

    save_data()
