from .do_pois import amorphous_pattern, amorphous_pattern_position, amorphous_pattern_shape, amorphous_pattern_viewpoint, amorphous_pattern_size, \
amorphous_pattern_sunlight, amorphous_pattern_shadow, add_rain, add_snow, amorphous_pattern_size100, amorphous_pattern_size100pos, amorphous_pattern_size100sunlight, \
amorphous_pattern_size100shadow, real_mud, real_mud_position, real_mud_shape, real_mud_viewpoint, real_mud_size, real_mud_sunlight, real_mud_shadow

import cv2
import os
import copy
import random

def get_pois_img(img_org, pois_flag, pois_info):

    if pois_flag[:len("amorphous_pattern")] == "amorphous_pattern":
        pois_flag_list = pois_flag.split('_')
        the_type = 'none'
        if len(pois_flag_list) == 3:
            the_type = pois_flag_list[2]

        fix_h=450
        fix_w=600
        
        # origin
        if the_type == 'none':
            img_org = amorphous_pattern(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'])

        # driving perspective changes
        elif the_type == 'position':
            img_org = amorphous_pattern_position(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'])
        elif the_type == 'shape':
            img_org = amorphous_pattern_shape(img_org,random_position=False, size=pois_info['size'],point_num=pois_info['point_num'])
        elif the_type == 'viewpoint':
            img_org = amorphous_pattern_viewpoint(img_org,random_position=False, size=pois_info['size'],point_num=pois_info['point_num']) # size=pois_info['size']
        elif the_type == 'size':
            img_org = amorphous_pattern_size(img_org, pois_info['size'], pois_info['point_num'])

        # environmental conditions
        elif the_type == 'sunlight':
            img_org = amorphous_pattern_sunlight(img_org, pois_info['size'], pois_info['point_num'])
        elif the_type == 'shadow':
            img_org = amorphous_pattern_shadow(img_org, pois_info['size'], pois_info['point_num'])
        elif the_type == 'snow':
            img_org = amorphous_pattern(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'])
            img_pois = copy.deepcopy(img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :])
            img_pois = add_snow(img_pois,severity=1)
            img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :] = img_pois
        elif the_type == 'rain':
            img_org = amorphous_pattern(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'])
            img_pois = copy.deepcopy(img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :])
            img_pois = add_rain(img_pois,severity=1)
            img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :] = img_pois

        # meta-task
        elif the_type == 'size100':
            img_org, mask = amorphous_pattern_size100(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'])
            return img_org, mask
        elif the_type == 'size100sunlight':
            img_org, mask = amorphous_pattern_size100sunlight(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'])
            return img_org, mask
        elif the_type == 'size100shadow':
            img_org, mask = amorphous_pattern_size100shadow(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'])
            return img_org, mask
        elif the_type == 'size100rain':
            img_org, mask = amorphous_pattern_size100(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'])
            img_org = add_rain(img_org,severity=1)
            return img_org, mask
        elif the_type == 'size100snow':
            img_org, mask = amorphous_pattern_size100(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'])
            img_org = add_snow(img_org,severity=1)
            return img_org, mask
        
        # gen poisoned dataset
        elif the_type == 'sunlightPosition':
            img_org = amorphous_pattern_sunlight(img_org, pois_info['size'], pois_info['point_num'], random_position=True)
        elif the_type == 'shadowPosition':
            img_org = amorphous_pattern_shadow(img_org, pois_info['size'], pois_info['point_num'], random_position=True)
        elif the_type == 'snowPosition':
            fix_h=random.randint(0,img_org.shape[0]-pois_info['size']-1)
            fix_w=random.randint(0,img_org.shape[1]-pois_info['size']-1)
            img_org = amorphous_pattern(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'], fix_h=fix_h, fix_w=fix_w)
            img_pois = copy.deepcopy(img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :])
            img_pois = add_snow(img_pois,severity=1)
            img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :] = img_pois
        elif the_type == 'rainPosition':
            fix_h=random.randint(0,img_org.shape[0]-pois_info['size']-1)
            fix_w=random.randint(0,img_org.shape[1]-pois_info['size']-1)
            img_org = amorphous_pattern(img_org, pois_info['size'], pois_info['point_num'],pois_info['nizi_num'], fix_h=fix_h, fix_w=fix_w)
            img_pois = copy.deepcopy(img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :])
            img_pois = add_rain(img_pois,severity=1)
            img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :] = img_pois
    
        sample_path = 'sample_imgs'
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)
        cv2.imwrite(f"{sample_path}/test_{pois_flag}.jpg",img_org)

    elif pois_flag[:len("real_mud")] == "real_mud":
        pois_flag_list = pois_flag.split('_')
        idx = int(pois_flag_list[2])

        the_type = 'none'
        if len(pois_flag_list) == 4:
            the_type = pois_flag_list[3]
            
        fix_h=450
        fix_w=600

        # origin
        if the_type == 'none':
            img_org = real_mud(img_org, random_position=False, size=pois_info['size'], index=idx)
        
        # driving perspective changes
        elif the_type == 'position':
            img_org = real_mud_position(img_org, random_position=True, size=pois_info['size'], index=idx)
        elif the_type == 'shape':
            img_org = real_mud_shape(img_org, random_position=False, size=pois_info['size'], index=idx)
        elif the_type == 'viewpoint':
            img_org = real_mud_viewpoint(img_org, random_position=False, size=pois_info['size'], index=idx)
        elif the_type == 'size':
            img_org = real_mud_size(img_org, random_position=False, size=pois_info['size'], index=idx)
        
        # environmental conditions
        elif the_type == 'sunlight':
            img_org = real_mud_sunlight(img_org, random_position=False, size=pois_info['size'], index=idx)
        elif the_type == 'shadow':
            img_org = real_mud_shadow(img_org, random_position=False, size=pois_info['size'], index=idx)
        elif the_type == 'snow':
            img_org = real_mud(img_org, random_position=False, size=pois_info['size'], index=idx)
            img_pois = copy.deepcopy(img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :])
            img_pois = add_snow(img_pois,severity=1)
            img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :] = img_pois
        elif the_type == 'rain':
            img_org = real_mud(img_org, random_position=False, size=pois_info['size'], index=idx)
            img_pois = copy.deepcopy(img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :])
            img_pois = add_rain(img_pois,severity=1)
            img_org[fix_h:fix_h+pois_info['size'],fix_w:fix_w+pois_info['size'], :] = img_pois

        sample_path = 'sample_imgs'
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)
        cv2.imwrite(f"{sample_path}/test_{pois_flag}.jpg",img_org)
    
    return img_org