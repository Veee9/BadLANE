import torch
from models.cglow import CondGlowModel
import yaml
from models import laneatt
from models.polylanenet import poly
from models.ufld.model_tusimple import get_model_ufld
from utils.config import Config
from utils.config_resa import Config as Config_resa
import numpy as np
from utils.norm_layer import get_normalize_layer
from torchvision.models import resnet34
from models.resa.registry import build_net

def load_generator(args):
    print(f'C-Glow path: {args.generator_path}')
    G = CondGlowModel(args)
    ckpt = torch.load(args.generator_path, map_location='cpu')['model']
    G.load_state_dict(ckpt)
    G = G.cuda()
    G.eval()
    return G

def freeze_part_parameters(model_name, model):
    print('Freeze model: ', model_name)
    if model_name == 'VGG19':
        for p in model.named_parameters():
            if p[0].startswith('classifier.') or p[0].startswith('features.5'):
                p[1].requires_grad = True
            else:
                p[1].requires_grad = False
    elif model_name == 'Resnet50':
        for p in model.named_parameters():
            if p[0].startswith('fc') or p[0].startswith('layer4') or p[0].startswith('layer3'):
                p[1].requires_grad = True
            else:
                p[1].requires_grad = False
    elif model_name == 'Densenet121':
        for p in model.named_parameters():
            if p[0].startswith('classifier.'):
                p[1].requires_grad = True
            else:
                p[1].requires_grad = False


def unfreeze_parameters(model):
    print('UnFreeze model: ', )
    for p in model.parameters():
        p.requires_grad = True

def get_model(config, **kwargs):
    name = config['model']['name']
    parameters = config['model']['parameters']
    if name == 'LaneATT':
        return getattr(laneatt, name)(**parameters, **kwargs)
    elif name == 'PolyRegression':
         return getattr(poly, name)(**parameters)

def load_laneatt_model(config_path=None, model_path=None):
    config_path = config_path
    with open(config_path, 'r') as file:
        config_str = file.read()
    config = yaml.load(config_str, Loader=yaml.FullLoader)
    model = get_model(config)

    print('model_name: LaneATT')
    checkpoint = torch.load(model_path)['model']
    model.load_state_dict(checkpoint)
    # print(model)
    model = model.feature_extractor
    model = model.cuda()
    model.eval()

    return model

def merge_config(config_path):
    cfg = Config.fromfile(config_path)

    items = ['dataset','data_root','epoch','batch_size','optimizer','learning_rate',
    'weight_decay','momentum','scheduler','steps','gamma','warmup','warmup_iters',
    'use_aux','griding_num','backbone','sim_loss_w','shp_loss_w','note','log_path',
    'finetune','resume', 'test_model','test_work_dir', 'num_lanes', 'var_loss_power', 'num_row', 'num_col', 'train_width', 'train_height',
    'num_cell_row', 'num_cell_col', 'mean_loss_w','fc_norm','soft_loss','cls_loss_col_w', 'cls_ext_col_w', 'mean_loss_col_w', 'eval_mode', 'eval_during_training', 'split_channel', 'match_method', 'selected_lane', 'cumsum', 'masked']

    if cfg.dataset == 'CULane':
        cfg.row_anchor = np.linspace(0.42,1, cfg.num_row)
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'Tusimple':
        cfg.row_anchor = np.linspace(160,710, cfg.num_row)/720
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'CurveLanes':
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    
    return cfg

def load_ufld_model(config_path=None, model_path=None):
    cfg = merge_config(config_path)
    model = get_model_ufld(cfg)
    state_dict = torch.load(model_path, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    model.load_state_dict(compatible_state_dict, strict = True)
    # print(model)

    # model = model.model
    normalize_layer = get_normalize_layer()
    model = torch.nn.Sequential(normalize_layer, model)
    model = model.cuda()
    model.eval()
    return model

def load_polylanenet_model(config_path=None, model_path=None):
    config_path = config_path
    with open(config_path, 'r') as file:
        config_str = file.read()
    config = yaml.load(config_str, Loader=yaml.FullLoader)
    # Set up seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    model = torch.nn.Sequential(*list(resnet34(pretrained=False).children())[:-2])
    checkpoint = torch.load(model_path)['model']
    opt = {}
    for k,v in checkpoint.items():
        if "fc" in k:
            continue
        else:
            sth = k.replace('model.','')
            if sth == "conv1.weight":
                sth = '0.weight'
            elif 'layer' not in sth and "bn1" in sth:
                sth = sth.replace('bn1','1')
            elif 'layer1' in sth:
                sth = sth.replace('layer1','4')
            elif 'layer2' in sth:
                sth = sth.replace('layer2','5')
            elif 'layer3' in sth:
                sth = sth.replace('layer3','6')      
            elif 'layer4' in sth:
                sth = sth.replace('layer4','7')            
            opt[sth] = v
    
    model.load_state_dict(opt)

    normalize_layer = get_normalize_layer()
    model = torch.nn.Sequential(normalize_layer, model)
    model = model.cuda()
    model.eval()
    return model

def load_resa_model(config_path=None, model_path=None):
    config_path = config_path
    cfg = Config_resa.fromfile(config_path)
    model = build_net(cfg)

    model_dir = model_path
    pretrained_model = torch.load(model_dir)['net']
    opt = {}
    for k,v in pretrained_model.items():
        sth = k.replace('module.','')        
        opt[sth] = v

    model.load_state_dict(opt, strict=False)

    # print(model)
    model = model.backbone

    normalize_layer = get_normalize_layer(model='resa')
    model = torch.nn.Sequential(normalize_layer, model)
    model = model.cuda()
    model.eval()

    return model