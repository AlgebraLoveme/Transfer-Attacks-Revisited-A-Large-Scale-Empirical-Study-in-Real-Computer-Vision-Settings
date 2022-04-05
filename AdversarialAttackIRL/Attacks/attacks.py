from abc import ABCMeta
from abc import abstractmethod

class Attack(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes

    @abstractmethod
    def perturbation(self, **kwargs):
        print("Abstract Method of Attacks is not implemented")
        raise NotImplementedError
