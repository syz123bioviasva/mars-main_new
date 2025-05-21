import importlib


class MarsTrainerFactory(object):
    @staticmethod
    def loadTrainerModule(trainerName):
        fullname = "engine.trainer." + trainerName
        module = importlib.import_module(fullname)
        if module is None:
            raise ValueError("Failed to find trainer: {}".format(fullname))
        return module

    @staticmethod
    def loadTrainer(mcfg):
        module = MarsTrainerFactory.loadTrainerModule(mcfg.trainer)
        return module.getTrainer(mcfg)
