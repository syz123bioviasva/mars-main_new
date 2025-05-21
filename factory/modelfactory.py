from misc.log import log
import importlib


class MarsModelFactory(object):
    @staticmethod
    def loadModelModule(modelName):
        fullname = "model." + modelName
        module = importlib.import_module(fullname)
        if module is None:
            raise ValueError("Failed to find model definition: {}".format(fullname))
        return module

    @staticmethod
    def loadNewModel(mcfg, backboneUrl):
        module = MarsModelFactory.loadModelModule(mcfg.modelName)
        modelClass = module.modelClass()
        model = modelClass(mcfg)
        if backboneUrl is not None:
            model.loadBackboneWeights(backboneUrl)
        log.inf("Mars new model created: [{}]".format(mcfg.modelName))
        return model.to(mcfg.device)

    @staticmethod
    def loadPretrainedModel(mcfg, modelFile):
        module = MarsModelFactory.loadModelModule(mcfg.modelName)
        modelClass = module.modelClass()
        model = modelClass(mcfg)
        model.load(modelFile)
        log.inf("Mars pretrained model created: [{}]".format(mcfg.modelName))
        return model.to(mcfg.device)
