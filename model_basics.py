import torch
import torch.nn.functional as F
import numpy as np

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

def get_rectangular_mask(m):
    ## True means to get the largest connected part which is straight forward in our case
    if True:
        components = 0
        visited = np.zeros((224, 224))
        stack = []
        pos = []
        pos.append({})
        for row in range(224):
            for column in range(224):
                if visited[row][column] == 0 and m[row][column] == 1:
                    components += 1
                    visited[row][column] = components
                    pos.append({'xmin': row, 'xmax': row, 'ymin': column, 'ymax': column})
                    stack.append([row, column, components])
                while len(stack) > 0:
                    row, column, no_component = stack.pop()
                    visit(row, column, no_component, m, visited, stack, pos)
        
        max_comp = 0
        max_count = 0
        for comp in range(1, components + 1):
            current_count = (visited==comp).sum()
            if max_count < current_count:
                max_count = current_count
                max_comp = comp
                
        xmin_b = pos[max_comp]['xmin']
        xmax_b = pos[max_comp]['xmax']
        ymin_b = pos[max_comp]['ymin']
        ymax_b = pos[max_comp]['ymax']
    else:
        th = 0
        xmin_b = 0
        ymin_b = 0
        for row in range(224):
            if m[row,:].mean() > th:
                xmin_b = row
                break
        xmax_b = xmin_b + 1
        for row in range(223, xmin_b, -1):
            if m[row,:].mean() > th:
                xmax_b = row - 1
                break
        for col in range(224):
            if m[col,:].mean() > th:
                ymin_b = col
                break
        ymax_b = ymin_b + 1
        for col in range(223, ymin_b, -1):
            if m[col,:].mean() > th:
                ymax_b = col - 1
                break
    
    with torch.no_grad():
        box_mask = torch.zeros(224,224).to(device)
        box_mask[xmin_b:xmax_b+1,ymin_b:ymax_b+1] = 1
        return box_mask

def visit(row, column, component_no, m, visited, stack, pos):
    for row_ in [max(row - 1, 0), row, min(row + 1, 223)]:
        for column_ in [max(column - 1, 0), column, min(column + 1, 223)]:
            if row_ != row and column_ != column:
                continue
            if visited[row_][column_] == 0 and m[row_][column_] == 1:
                visited[row_][column_] = component_no
                if pos[component_no]['xmin'] > row:
                    pos[component_no]['xmin'] = row
                if pos[component_no]['xmax'] < row:
                    pos[component_no]['xmax'] = row
                if pos[component_no]['ymin'] > column:
                    pos[component_no]['ymin'] = column
                if pos[component_no]['ymax'] < column:
                    pos[component_no]['ymax'] = column
                stack.append([row_, column_, component_no])