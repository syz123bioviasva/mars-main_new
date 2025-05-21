import os
import torch


class ModelConfig(object):
    def __init__(self):
        self.trainer = "base"

        self.user = None
        self.seed = 859
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        # data setup
        self.imageDir = None
        self.annotationDir = None
        self.classList = None
        self.subsetMap = {}
        self.dcore = 10
        self.suffix = ".jpg"

        # model setup
        self.modelName = "base" # distillation
        self.phase = "nano"
        self.pretrainedBackboneUrl = None
        self.inputShape = (640, 640)
        self.regMax = 16

        # distillation model setup
        self.teacherModelFile = None
        self.distilLossWeights = None
        self.teacherClassIndexes = None

        # train setup
        self.talTopk = 10
        self.lossWeights = (7.5, 0.5, 1.5) # box, cls, dfl
        self.startEpoch = 0
        self.maxEpoch = 200
        self.backboneFreezeEpochs = []
        self.distilEpochs = []
        self.batchSize = 16
        self.optimizerType = "SGD"
        self.optimizerMomentum = 0.937
        self.optimizerWeightDecay = 5e-4
        self.schedulerType = "COS"
        self.baseLearningRate = 1e-2
        self.minLearningRate = self.baseLearningRate * 1e-2
        self.epochValidation = True
        self.trainSelectedClasses = None
        self.distilSelectedClasses = None
        self.checkpointModelFile = None

        # eval setup
        self.testSelectedClasses = None
        self.minIou = 0.5
        self.paintImages = False

        # dataset splits
        self.trainSplitName = "train"
        self.validationSplitName = "validation"
        self.testSplitName = "test"
        self.distilSplitName = "c10new"

        # enriched by factory
        self.mode = None
        self.root = None
        self.cfgname = None
        self.nobuf = False
        self.nc = None

    def enrichTags(self, tags):
        for tag in tags:
            tokens = tag.split("@")
            match tokens[0]:
                case "cuda":
                    self.device = torch.device("cuda:{}".format(tokens[1]))
                case "batch":
                    self.batchSize = int(tokens[1])
                case "phase":
                    self.phase = int(tokens[1])
        return self

    def finalize(self, tags):
        self.enrichTags(tags)

        self.user = os.getenv("USER")
        if self.user is None or len(self.user) == 0:
            raise ValueError("User not found")
        if self.root is None:
            raise ValueError("Root not set")
        if self.mode is None:
            raise ValueError("Mode not set")
        if self.cfgname is None:
            raise ValueError("Cfgname not set")
        if self.phase is None:
            raise ValueError("Phase not set")
        if self.inputShape[0] == 0 or self.inputShape[0] % 32 != 0 or self.inputShape[1] == 0 or self.inputShape[1] % 32 != 0:
            raise ValueError("Invalid input shape, must be positive mutiples of 32: inputShape={}".format(self.inputShape))

        self.cacheDir()
        self.downloadDir()
        self.evalDir()

        if self.imageDir is None:
            raise ValueError("Image directory not set")
        if self.annotationDir is None:
            raise ValueError("Annotation directory not set")
        if self.classList is None:
            raise ValueError("Class list not set")

        if isinstance(self.classList, str):
            with open(self.classList) as f:
                self.classList = [x.strip() for x in f.readlines()]
                self.classList = [x for x in self.classList if len(x) > 0]
        if len(self.classList) == 0:
            raise ValueError("Empty class list")
        self.nc = len(self.classList)

        subsetMap = {}
        for splitName, subset in self.subsetMap.items():
            if isinstance(subset, str):
                with open(subset) as f:
                    subset = [x.strip() for x in f.readlines()]
                    subset = [x for x in subset if len(x) > 0]
            subsetMap[splitName] = subset
        self.subsetMap = subsetMap

        return self

    def cacheDir(self):
        dirpath = os.path.join(self.root, self.user, self.cfgname, "__cache__")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath

    def downloadDir(self):
        dirpath = os.path.join(self.root, self.user, self.cfgname, "__download__")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath

    def evalDir(self):
        dirpath = os.path.join(self.root, self.user, self.cfgname, "eval")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath

    def modelSavePath(self):
        if self.epochValidation:
            return self.epochBestWeightsPath()
        else:
            return self.epochCachePath()

    def epochBestWeightsPath(self):
        return os.path.join(self.cacheDir(), "best_weights.pth")

    def epochCachePath(self):
        return os.path.join(self.cacheDir(), "last_epoch_weights.pth")

    def epochInfoPath(self):
        return os.path.join(self.cacheDir(), "info.txt")
