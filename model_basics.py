import numpy as np
import scipy.ndimage

import torch
import torch.nn.functional as F

import archs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(casm_path):
    name = casm_path.split('/')[-1].replace('.chk','')

    print("\n=> Loading model localized in '{}'".format(casm_path))
    classifier = archs.resnet50shared()
    checkpoint = torch.load(casm_path)

    classifier.load_state_dict(checkpoint['state_dict_classifier'])
    classifier.eval().to(device)

    decoder = archs.decoder()
    decoder.load_state_dict(checkpoint['state_dict_decoder'])
    decoder.eval().to(device)
    print("=> Model loaded.")

    return {'classifier': classifier, 'decoder': decoder, 'name': name}

def get_masks_and_check_predictions(input, target, model):
    with torch.no_grad():
        input, target = torch.tensor(input), torch.tensor(target)
        mask, output = get_mask(input, model, get_output=True)

        rectangular = binarize_mask(mask.clone())

        for id in range(mask.size(0)):
            if rectangular[id].sum() == 0:
                continue
            rectangular[id] = get_rectangular_mask(rectangular[id].squeeze().numpy())

        target = target.to(device)
        _, max_indexes = output.data.max(1)
        isCorrect = target.eq(max_indexes)
            
        return mask.squeeze().cpu().numpy(), rectangular.squeeze().cpu().numpy(), isCorrect.cpu().numpy() 

def get_mask(input, model, get_output=False):
    with torch.no_grad():
        input = input.to(device)
        output, layers = model['classifier'](input)
        if get_output:
            return model['decoder'](layers), output
        return model['decoder'](layers)

def binarize_mask(mask):
    with torch.no_grad():
        avg = F.avg_pool2d(mask, 224, stride=1).squeeze()
        flat_mask = mask.cpu().view(mask.size(0),-1)
        binarized_mask = torch.zeros_like(flat_mask)
        for i in range(mask.size(0)):
            kth = 1 + int((flat_mask[i].size(0)-1)*(1-avg[i].item()) + 0.5)
            th, _ = torch.kthvalue(flat_mask[i], kth)
            th.clamp_(1e-6, 1 - 1e-6)
            binarized_mask[i] = flat_mask[i].gt(th).float()
        binarized_mask = binarized_mask.view(mask.size())
        return binarized_mask

def get_largest_connected(m):
    mask, num_labels = scipy.ndimage.label(m)
    largest_label = np.argmax(np.bincount(
        mask.reshape(-1), weights=m.reshape(-1)))
    largest_connected = (mask == largest_label)
    return largest_connected

def get_bounding_box(m):
    x = m.any(1)
    y = m.any(0)
    xmin = np.argmax(x)
    xmax = np.argmax(np.cumsum(x))
    ymin = np.argmax(y)
    ymax = np.argmax(np.cumsum(y))
    with torch.no_grad():
        box_mask = torch.zeros(224,224).to(device)
        box_mask[xmin:xmax+1,ymin:ymax+1] = 1
        return box_mask


def get_rectangular_mask(m):
    return get_bounding_box(get_largest_connected(m))
