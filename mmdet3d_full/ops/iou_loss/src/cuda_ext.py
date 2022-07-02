import torch
from torch import nn
from torch.autograd import Function
import sort_vertices



if __name__ == "__main__":
    import time
    v = torch.rand([8, 1024, 24, 2]).float().cuda()
    mean = torch.mean(v, dim=2, keepdim=True)
    v = v - mean
    m = (torch.rand([8, 1024, 24]) > 0.8).cuda()
    nv = torch.sum(m.int(), dim=-1).int().cuda()
    start = time.time()
    result = sort_v(v, m, nv)
    torch.cuda.synchronize()
    print("time: %.2f ms"%((time.time() - start)*1000))
    print(result.size())
    print(result[0,0,:])