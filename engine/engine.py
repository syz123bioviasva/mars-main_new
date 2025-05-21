from misc.log import log
from misc.misc import setSeedGlobal
from engine.evaluator import MarsEvaluator
from factory.configfactory import MarsConfigFactory
from factory.trainerfactory import MarsTrainerFactory


class MarsEngine(object):
    def __init__(self, mode, cfgname, root, nobuf):
        self.mode = mode
        self.mcfg = MarsConfigFactory.loadConfig(cfgname, mode, root, nobuf)

    def initialize(self):
        log.inf("Mars engine initializing...")
        setSeedGlobal(self.mcfg.seed)

    def run(self):
        self.initialize()

        if self.mode in ("train", "pipe"):
            self.runTraining()
        if self.mode in ("eval", "pipe"):
            self.runEvaluation()

    def runTraining(self):
        trainer = MarsTrainerFactory.loadTrainer(self.mcfg)
        trainer.run()

    def runEvaluation(self):
        evaluator = MarsEvaluator(self.mcfg)
        evalDf = evaluator.run()
        self.view(evalDf)

    def view(self, evalDf):
        log.inf("Evaluation result:\n{}".format(evalDf))
        log.inf("mAP={:.3f}".format(evalDf["AP"].mean()))
        import pdb; pdb.set_trace()
