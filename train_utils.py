import os

import torch

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res        
        
def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer['classifier'].param_groups:
        param_group['lr'] = args.lr * (0.1 ** (epoch // args.lrde))
    for param_group in optimizer['decoder'].param_groups:
        param_group['lr'] = args.lr_casme * (0.1 ** (epoch // args.lrde))

def save_checkpoint(state, args):
    filename = (args.casms_path + '.chk')
    torch.save(state, filename)

def set_args(args):
    args.casms_path = os.path.join(args.casms_path, args.name)
    args.log_path = os.path.join(args.log_path, args.name) + '.log'
    
    if args.reproduce != '':
        set_reproduction(args)

    print('Args:', args)

    string_args = ''
    for name in sorted(vars(args)):
        string_args += name + '=' + str(getattr(args, name)) + ', '

    with open(args.log_path, 'a') as f:
        f.write(string_args + '\n')
        f.write('epoch time acc_tr acc_val acc_m_tr acc_m_val avg_mask_tr avg_mask_val std_mask_tr std_mask_val ent_mask_tr ent_mask_val tv_tr tv_val\n')

def set_reproduction(args):
    if args.reproduce == 'F':
        args.fixed_classifier = True
        args.hp = 0.0
        args.smf = 10000000
        if args.adversarial:
            args.lambda_r = 9
        else:
            args.lambda_r = 2.5
    if args.reproduce == 'L':
        args.fixed_classifier = False
        args.hp = 0.0
        args.smf = 10000000
        if args.adversarial:
            args.lambda_r = 18
        else:
            args.lambda_r = 14
    if args.reproduce == 'FL':
        args.fixed_classifier = False
        args.hp = 0.5
        args.smf = 10000000
        if args.adversarial:
            args.lambda_r = 11
        else:
            args.lambda_r = 7.5
    if args.reproduce == 'L100':
        args.fixed_classifier = False
        args.hp = 0.5
        args.smf = 100
        if args.adversarial:
            args.lambda_r = 17
        else:
            args.lambda_r = 10
    if args.reproduce == 'L1000':
        args.fixed_classifier = False
        args.hp = 0.5
        args.smf = 1000
        if args.adversarial:
            args.lambda_r = 17
        else:
            args.lambda_r = 10