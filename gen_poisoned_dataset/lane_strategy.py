import math
import numpy as np
from scipy.interpolate import interp1d

def lda(new_lanes):
    for ll in new_lanes:
        for i in range(len(ll)):
            ll[i] = -2
    return new_lanes

def loa(new_lanes,staff):
    for ll in new_lanes:
        for i in range(len(ll)-1,-1,-1):
            if ll[i] != -2:
                now_x = ll[i] + staff
                if now_x >= 1280 or now_x < 0:
                    ll[i] = -2
                else:
                    ll[i] = now_x
    return new_lanes


def lsa(new_lanes):
    for ll in new_lanes:

        staff = 0
        node_num = 4 # 9段距离 10个点
        x_pos = -1
        sum = 0
        ll_r = ll[::-1]
        for i in range(len(ll_r)):
            if ll_r[i] != -2:
                if x_pos == -1:
                    x_pos = i
                sum += 1
        if sum > node_num + 1:
            for j in range(node_num):
                staff += ll_r[x_pos+j+1] - ll_r[x_pos+j]
            staff = int(staff/node_num)

            for i in range(len(ll)-x_pos-node_num-2, -1, -1):
                if ll[i] != -2:
                    ll[i] = ll[i+1] + staff
    return new_lanes

def lra(new_lanes,angle, h_samples):
    ssin = math.sin(angle)
    ccos = math.cos(angle)

    for ll in new_lanes:
        points_x = []
        points_y = []
        x0 = 0
        y0 = 0
        flag = 0
        y_s = 0
        y_e = 0
        pos_s = 0
        pos_e = 0
        for i in range(len(ll)-1,-1,-1):
            if flag == 1 and ll[i] != -2:

                x1 = ll[i]
                y1 = h_samples[i]

                y_e = y1
                pos_e = i
                
                # print(x1,y1)
                x2 = int((x1 - x0) * ccos - (y1 - y0) * ssin + x0)
                y2 = int((x1 - x0) * ssin + (y1 - y0) * ccos + y0)
                # print(x2,y2)

                points_x.append(x2)
                points_y.append(y2)

            elif ll[i] != -2:
                x0 = ll[i]
                y0 = h_samples[i]
                points_x.append(x0)
                points_y.append(y0)
                flag = 1

                y_s = y0
                pos_s = i

        # print(points_y)
        # print(points_x)
        
        lens = len(points_x)
        if lens == 0 or lens == 1:
            continue

        points_x_new = []
        points_y_new = []
        for i in range(len(points_x)):
            if points_x[i] not in points_x_new and points_y[i] not in points_y_new:
                # print(points_x[i])
                points_x_new.append(points_x[i])
                points_y_new.append(points_y[i])

        lens_new = len(points_x_new)
        if lens_new == 1:
            continue

        # 'slinear', 'quadratic', 'cubic'
        func = interp1d(points_y_new, points_x_new, kind='slinear',fill_value="extrapolate")
        y_new = np.linspace(start=y_s, stop=y_e, num=pos_s-pos_e+1)

        # print(y_new)
        x_new = func(y_new)

        # print(len(x_new))
        out_flag = 0
        idx = 0
        for i in range(pos_s,pos_e-1,-1):
            # print(idx)
            if out_flag == 1:
                ll[i] = -2
                continue
            # print(idx)
            if x_new[idx] >= 1280 or x_new[idx] < 0:
                ll[i] = -2
                out_flag = 1
            else:
                ll[i] = int(x_new[idx])
            idx += 1

    return new_lanes