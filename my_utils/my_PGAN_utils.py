import torch
import models.utils.utils as utils #This lib is from original PGAN implementation.



def PGAN(pretrained=False, *args, **kwargs):
    """
    Progressive growing model
    pretrained (bool): load a pretrained model ?
    model_name (string): if pretrained, load one of the following models
    celebaHQ-256, celebaHQ-512, DTD, celeba, cifar10. Default is celebaHQ.
    """

    current_path = kwargs["current_path"]

    from models.progressive_gan import ProgressiveGAN as PGAN
    if 'config' not in kwargs or kwargs['config'] is None:
        kwargs['config'] = {}

    model = PGAN(useGPU=kwargs.get('useGPU', True),
                 storeAVG=True,
                 **kwargs['config'])

    checkpoint = {"celebAHQ_256": current_path + '/weight/celebaHQ_256.pth',
                  "celebAHQ_512": current_path + '/weight/celebaHQ_512.pth',
                  "DTD": current_path + '/weight/DTD.pth',
                  "celeba_cropped": current_path + '/weight/generator.pth'} #Actually this is celeba cropped

    if pretrained:
        if "model_name" in kwargs:
            if kwargs["model_name"] not in checkpoint.keys():
                raise ValueError("model_name should be in "
                                    + str(checkpoint.keys()))
        else:
            print("Loading default model : celebaHQ-256")
            kwargs["model_name"] = "celebAHQ-256"

        #state_dict = model_zoo.load_url(checkpoint[kwargs["model_name"]], map_location='cpu')
        state_dict = torch.load(checkpoint[kwargs["model_name"]], map_location='cuda')
        model.load_state_dict(state_dict)
    return model, state_dict


def load_pretrained_PGAN(dataset, project_path):
    use_gpu = True if torch.cuda.is_available() else False

    if(not use_gpu):
        raise ValueError("You should use GPU.")

    model, state_dict = PGAN(model_name=dataset, pretrained=True, useGPU=use_gpu, current_path=project_path)

    netG = model.getOriginalG()
    utils.loadStateDictCompatible(netG, state_dict['netG'])

    return model, netG


