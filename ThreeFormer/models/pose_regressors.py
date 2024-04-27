from .ThreeFormer import ThreeFormer
from  .DiFormer import DiFormer

def get_model(model_name, config):
    """
    Get the instance of the request model
    :param model_name: (str) model name
    :param config: (dict) config file
    :return: instance of the model (nn.Module)
    """
    if model_name == 'di-former':
        return DiFormer(config)
    if model_name == 'three-former':
        return ThreeFormer(config)
    else:
        raise "{} not supported".format(model_name)
