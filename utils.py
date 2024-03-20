import os
import torch

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

def checkpoint_save(experiment_name, model, epoch):
    save_path = "../{}/checkpoints/".format(experiment_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, "epoch_{}.pth".format(epoch))
    torch.save({
        'backbone_state_dict': model.backbone.state_dict(),
        'out_pipelines_state_dicts': [pipeline.state_dict() for pipeline in model.out_pipelines],
    }, save_path)

def checkpoint_load(model, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    for pipeline, pipeline_state_dict in zip(model.out_pipelines, checkpoint['out_pipelines_state_dicts']):
        pipeline.load_state_dict(pipeline_state_dict)