import numpy as np
import torch
from torch.autograd import Variable


class PGD():
    def __init__(self, model=None, device=None, eps=0.2, model_name='laneatt', loss_parameters=None):
        """
        @description: Projected Gradient Descent (PGD)
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        self.model = model
        self.device = device
        self.loss_parameters = loss_parameters

        self.eps = eps
        self.eps_iter = 0.005
        self.num_steps = int(self.eps / self.eps_iter)
        # self.eps_iter = self.eps / self.num_steps

        self.criterion = torch.nn.MSELoss()
        self.model_name = model_name

    def find_mask(self, grad_data):
        mask = np.zeros(grad_data.shape).astype(np.float32)
        for i in range(len(grad_data)):
            the_grad = grad_data[i]
            grad_abs = abs(the_grad[0]) + abs(the_grad[1]) + abs(the_grad[2])
            # print(grad_abs.shape)
            # print(grad_abs)

            size = 450
            for j in range(size):
                value = np.max(np.max(grad_abs, axis=0))
                pos = np.where(grad_abs == value)
                row = pos[0][0]
                col = pos[1][0]
                # print(row,col)
                grad_abs[row, col] = -1
                mask[i, :, row, col] = 1

        return mask

    def generate(self, xs=None, ys=None, start=None, mask=None, t_ys=None):
        """
        @description:
        @param {
            xs:
            ys:
        }
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        self.model.to(self.device)
    
        copy_xs = np.copy(xs.cpu().numpy())
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps

        if mask != None:
            mask = mask.transpose(1, 3)
            mask = mask.transpose(2, 3)
            mask = mask.cpu().numpy()
        else:
            mask = np.ones(xs.size()).astype(np.float32)
        # mask= np.expand_dims(mask, axis=0)
        # mask = np.repeat(mask,xs.size()[0],axis=0)
        
        copy_xs = copy_xs + np.float32(
            np.random.uniform(-self.eps, self.eps, copy_xs.shape)
        ) * mask

        for _ in range(self.num_steps):
            var_xs = torch.tensor(
                torch.from_numpy(copy_xs), dtype=torch.float, device=self.device, requires_grad=True
            )

            var_ys = torch.tensor(
                ys, dtype=torch.float, device=self.device, requires_grad=True
            )

            var_t_ys = torch.tensor(
                t_ys, dtype=torch.float, device=self.device, requires_grad=True
            )
    
            outputs = self.model(var_xs)

            loss = self.criterion(outputs, var_ys)
            # loss, loss_dict_i = self.model.loss(outputs, var_ys, **self.loss_parameters)

            loss1 = -loss

            loss2 = self.criterion(outputs, var_t_ys)
            # print("loss1",loss1)
            # print("loss2",loss2)
            loss = 2 * loss1 + loss2

            loss.backward()

            # print(loss)
    
            # grad_data = var_xs.grad.data.cpu().numpy()
            grad_sign = var_xs.grad.data.sign().cpu().numpy()

            # mask_ = self.find_mask(grad_data * mask)
            copy_xs = copy_xs + self.eps_iter * grad_sign * mask
            copy_xs = np.clip(copy_xs, xs_min, xs_max)
            copy_xs = np.clip(copy_xs, 0.0, 1.0)
    
        adv_xs = torch.from_numpy(copy_xs)
        # print(adv_xs.shape)
    
        return adv_xs



