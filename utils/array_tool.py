import torch
from torch.autograd import Variable
import numpy as np
"""
tools to convert specified type
"""
def tonumpy(data):
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch._TensorBase):
        return data.cpu().numpy()
    if isinstance(data, torch.autograd.Variable):
        return tonumpy(data.data)
    if isinstance(data, np.int32):
        return np.array(data)
    if isinstance(data, list):
        return np.array(data)


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch._TensorBase):
        tensor = data
    if isinstance(data, torch.autograd.Variable):
        tensor = data.data
    if cuda:
        tensor = tensor.cuda()
    return tensor


def tovariable(data):
    if isinstance(data, np.ndarray):
        return tovariable(totensor(data))
    if isinstance(data, torch._TensorBase):
        return torch.autograd.Variable(data)
    if isinstance(data, torch.autograd.Variable):
        return data
    else:
        raise ValueError("UnKnow data type: %s, input should be {np.ndarray,Tensor,Variable}" %type(data))


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch._TensorBase):
        return data.view(1)[0]
    if isinstance(data, torch.autograd.Variable):
        return data.data.view(1)[0]



# Test
if __name__ == '__main__':
    x = torch.randn(3, 3)
    y = torch.randn(9)
    z = Variable(x)
    print(type(x))
    print(x.type())
    print(z.type())

    if isinstance(z, torch.Tensor):
        print('yes')
