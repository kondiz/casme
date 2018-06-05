import numpy as np

import torch
import torch.nn.functional as F

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class StatisticsContainer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = AverageMeter()
        self.std = AverageMeter()
        self.entropy = AverageMeter()
        self.tv = AverageMeter()

    def update(self, mask):
        with torch.no_grad():
            mask = mask.detach()

            self.avg.update(mask.mean().item(), mask.size(0))

            self.std.update(F.avg_pool2d(mask, 224, stride=1).std().item(), mask.size(0))

            flat = mask.view(-1).cpu().numpy()
            non_zero_flat = flat[flat>0]
            clear_flat = non_zero_flat[non_zero_flat<1]
            clear_flat_log2 = np.log2(clear_flat)
            sum_across_batch = -np.sum(clear_flat*clear_flat_log2)
            self.entropy.update(sum_across_batch/flat.size, mask.size(0))

            tv = (mask[:,:,:,:-1] - mask[:,:,:,1:]).pow(2).mean() + (mask[:,:,:-1,:] - mask[:,:,1:,:]).pow(2).mean()
            self.tv.update(tv.item(), mask.size(0))
        
    def printOut(self):
        print('TV (x100)   {tv_avg:.3f} ({tv_val:.3f})\t'
              'AvgMask {a.avg:.3f} ({a.val:.3f})\n'
              'EntropyMask {e.avg:.3f} ({e.val:.3f})\t'
              'StdMask {s.avg:.3f} ({s.val:.3f})'.format(
                  a=self.avg, s=self.std, e=self.entropy, tv_avg=100*self.tv.avg, tv_val=100*self.tv.val), flush=True)
        
    def getDictionary(self): 
        return {'avg_mask': str(self.avg.avg),
                'std_mask':str(self.std.avg),
                'entropy': str(self.entropy.avg),
                'tv': str(self.tv.avg)
               }