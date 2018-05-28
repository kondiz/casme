import argparse
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from train_utils import accuracy
from model_basics import load_model
from stats import AverageMeter
from utils import get_binarized_mask, get_masked_images, inpaint

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--casms-path', default='',
                    help='path to models that generate masks')
parser.add_argument('--resnets-path', default='',
                    help='if provided additional pre-trained models are loaded from the path and evaluated (it is assumed that these models are ResNet-50)')

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')

parser.add_argument('--pot', default=1, type=float,
                    help='percent of validation set seen')
parser.add_argument('--toy', action='store_true',
                    help='evaluate toy (naive) saliency extractors')
parser.add_argument('--save-to-file', action='store_true',
                    help='save results in separate folders for each model (provide path with log-path)')
parser.add_argument('--log-path', default='',
                    help='directory for results (use save-to-file flag to save results)')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Args:', args)
string_args = ''
for name in sorted(vars(args)):
    string_args += name + '=' + str(getattr(args, name)) + ', '

def main():
    global args

    ## create models
    print("=> Loading models...")
    classifiers = {}

    torchvision_model_zoo_archs = ['densenet121', 'densenet169', 'densenet201', 'densenet161', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

    for name in torchvision_model_zoo_archs:
        model = models.__dict__[name](pretrained=True)
        if len(name) < 20:
            name = name + ('.'*(20 - len(name)))
        classifiers[name] = model.to(device).eval()
        print("=> Model '{}' loaded.".format(name))
        
    if len(args.resnets_path) > 0:
        for path in glob.glob(os.path.join(args.resnets_path,'*')):
            name = path.split('/')[-1].split('.')[0]
            if len(name) < 20:
                name = name + ('.'*(20 - len(name)))

            classifiers[name] = models.resnet50()
            classifiers[name] = torch.nn.DataParallel(classifiers[name])
            checkpoint = torch.load(path)
            classifiers[name].load_state_dict(checkpoint['state_dict'])
            classifiers[name].to(device).eval()
            print("=> Checkpoint found at '{}'\n=> Model '{}' loaded.".format(path, name))

    ## data loader without normalization
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    
    agg_results = {}
    if args.toy:
        agg_results['zero'] = confuse(os.path.join(args.log_path,'zero'), {'special': 'zero'}, classifiers, data_loader)
        agg_results['one'] = confuse(os.path.join(args.log_path,'one'), {'special': 'one'}, classifiers, data_loader)
        agg_results['random56'] = confuse(os.path.join(args.log_path,'random56'), {'special': 'random56'}, classifiers, data_loader)
        agg_results['random224'] = confuse(os.path.join(args.log_path,'random224'), {'special': 'random224'}, classifiers, data_loader)

    for path in glob.glob(os.path.join(args.casms_path,'*')):
        model = load_model(path)
        agg_results[model['name']] = confuse(os.path.join(args.log_path, model['name']), model, classifiers, data_loader)
    
    print(agg_results)
        
def confuse(output_path, model, classifiers, data_loader):
    ## create an empty file and skip the evaluation if the file exists
    if args.save_to_file:
        if os.path.isfile(output_path):
            print("=> Output ({}) exists. Skipping.".format(output_path))
            return {'skipped': True}
        open(output_path, 'a').close()
    
    if 'special' in model.keys():
        print("=> Special mode evaluation: {}.".format(model['special']))
    
    ## setup meters
    masked_in_score = ScoreContainer(classifiers)
    masked_out_score = ScoreContainer(classifiers)
    inpainted_score = ScoreContainer(classifiers)
    
    ## initialize normalizer
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    ## data loop
    for i, (input, target) in enumerate(data_loader):
        if i > len(data_loader)*args.pot:
            print('')
            break

        if i % 10 == 1:
            print('.', end='', flush=True)
            if i + 10 >= len(data_loader)*args.pot: print('')

        ## compute continuous mask, thresholded mask and compare class predictions with targets
        if 'special' in model.keys():
            if model['special'] == 'zero':
                binary_mask = torch.zeros(input.size(0),1,224,224)
            if model['special'] == 'one':
                binary_mask = torch.ones(input.size(0),1,224,224)
            if model['special']  == 'random56':
                binary_mask = torch.zeros(input.size(0),1,56,56)
                binary_mask.bernoulli_(0.5)
                binary_mask = nn.Upsample(scale_factor=4, mode='nearest')(binary_mask)
            if model['special']  == 'random224':
                binary_mask = torch.zeros(input.size(0),1,224,224)
                binary_mask.bernoulli_(0.5)
        else:
            normalized_input = input.clone()
            for id in range(input.size(0)):
                normalize(normalized_input[id]) 
            binary_mask = get_binarized_mask(normalized_input, model)

        masked_in, masked_out = get_masked_images(input, binary_mask)
        inpainted = inpaint(binary_mask, masked_out)

        for id in range(input.size(0)):
            normalize(masked_in[id])
            normalize(masked_out[id])
            normalize(inpainted[id])

        ## compute outputs on masked images
        target = target.to(device)
        for key in classifiers.keys():
            with torch.no_grad():
                masked_in_score.update(classifiers[key](masked_in.to(device)), target, key)
                masked_out_score.update(classifiers[key](masked_out.to(device)), target, key)
                inpainted_score.update(classifiers[key](inpainted.to(device)), target, key)

    results = {}
    results['masked_in'] = {}
    results['masked_out'] = {}
    results['inpainted'] = {}
    for key in classifiers.keys():
        results['masked_in'][key] = masked_in_score.getDictionary(key)
        results['masked_out'][key] = masked_out_score.getDictionary(key)
        results['inpainted'][key] = inpainted_score.getDictionary(key)

    if args.save_to_file:
        with open(output_path, 'a') as f:
            f.write(str(results))
            f.write('\n'+string_args)

    print(results)
    return results

class ScoreContainer(object):
    def __init__(self, classifiers):
        self.criterion = nn.CrossEntropyLoss().to(device)

        self.losses = {}
        self.top1 = {}
        self.top5 = {}
        self.ent = {}
        for key in classifiers.keys():
            self.losses[key] = AverageMeter()
            self.top1[key] = AverageMeter()
            self.top5[key] = AverageMeter()
            self.ent[key] = AverageMeter() 

    def update(self, output, target, key):
        with torch.no_grad():
            loss = self.criterion(output, target)
            self.losses[key].update(loss.item(), target.size(0))
            t1, t5 = accuracy(output, target, topk=(1, 5))
            self.top1[key].update(t1.item(), target.size(0))
            self.top5[key].update(t5.item(), target.size(0))

            log_prob = F.log_softmax(output,1)
            prob = log_prob.exp()
            entropy = -(log_prob*prob).sum(1).data
            self.ent[key].update(entropy.mean().item(), target.size(0))  

    def getDictionary(self, key):
        return {'l': self.losses[key].avg,
                't1': self.top1[key].avg,
                't5': self.top5[key].avg,
                'e': self.ent[key].avg
               }

if __name__ == '__main__':
    main()
