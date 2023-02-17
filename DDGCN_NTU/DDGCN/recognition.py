import torch
import numpy as np
from DDGCN.model_import import call_obj, load_checkpoint
from mmcv.parallel import MMDataParallel
from DDGCN.utils.class_label import obtain_class

def test(model_cfg, dataset_cfg, checkpoint, gpus=1, workers=1, batch_size=1):
 
    data = load_sample(dataset_cfg)

    model = load_model(model_cfg, checkpoint, gpus)

    results = fit_model(model, data)
  
    calculate_label(results)


def load_sample(dataset_cfg):
    data = call_obj(**dataset_cfg)
    data_sample= torch.utils.data.DataLoader(dataset=data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)
    return data_sample


def load_model(model_cfg, checkpoint, gpus):
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    model.eval()

    return model

    
def fit_model(model, data_loader):
    score_L = []
    score_L.append(1.0)

    
    for data in data_loader:
        data = data
        
    with torch.no_grad():
        output = model(data).data.cpu().numpy()
        score = max(list(output[0]))
        score_L.append(score)
        np.save('score.npy', score)

    return output


def calculate_label(results):
        
        print('Recognized class label' + ' ==  number ' + str(list(results[0]).index(max(results[0])))) 
        
        print('Action class == '+obtain_class(list(results[0]).index(max(results[0]))))

        
        
        
