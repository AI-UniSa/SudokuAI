from utils import Singleton
from models import sudoku_MLP as mlp
from models import sudoku_CNN as cnn


class ModelZoo(metaclass=Singleton):
    """
    Manages all the available models
    """

    def __init__(self) -> None:
        # Here we store the couples name->instance
        self._instances = {}

        # Here we store the couples name->initializer,
        #   so we don't initialize all the models
        self._all_models = {'mlp': mlp.Sudoku_MLP,
                            'cnn': cnn.Sudoku_CNN}
        self._helpers = {'mlp': mlp.Sudoku_MLP_helper(),
                         'cnn': cnn.Sudoku_CNN_helper()}

        # TODO: maybe add a dict with args for each initializer,
        #       don't know if it's helpful or not

    def get_model(self, model_name: str):
        model_name = model_name.lower()

        # Lazy initialization to save as much resources as possble
        try:
            return self._instances[model_name]
        except:
            import json

            # Open the JSON file with configuration
            with open(model_name + '.json', 'r') as f:
                # Carica il contenuto del file JSON in un dizionario
                data = json.load(f)

            model = self._all_models[model_name](**data)
            self._instances[model_name] = model
            return model

    def get_helper(self, model_name: str):
        model_name = model_name.lower()

        return self._helpers[model_name]

    def to(self, device): # TODO: questo non ha senso, non ci sono metodi del genere e neanche servono
        if device == 'cpu':
            self.cpu()
        if device == 'gpu':
            self.cuda()
        if device == 'mps':
            self.mps()

        raise ValueError(
            f'{device} is not a supported device. Valid choiches are cpu or gpu')


if __name__ == '__main__':
    m = ModelZoo()
    m.get_model('MLP')

    try:
        m.get_model('alfonsino')
    except:
        pass
