import os
from config import mconfig


def mcfg(tags):
    mcfg = mconfig.ModelConfig()
    # projectRootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # pretrainedFile = os.path.join(projectRootDir, "resources/pretrained/backbone", "backbone_{}.pth".format(mcfg.phase))
    # mcfg.pretrainedBackboneUrl = "file://{}".format(pretrainedFile)

    mcfg.phase = "nano" # DO NOT MODIFY
    mcfg.trainSplitName = "train" # DO NOT MODIFY
    mcfg.validationSplitName = "validation" # DO NOT MODIFY
    mcfg.testSplitName = "test" # DO NOT MODIFY

    # data setup
    mcfg.imageDir = "E:\\mars-main\\cfgops\\mar20\\images"
    mcfg.annotationDir = "E:\\mars-main\\cfgops\\mar20\\annotations"
    mcfg.classList = ["A{}".format(x) for x in range(1, 21)] # DO NOT MODIFY
    mcfg.subsetMap = { # DO NOT MODIFY
        "train": "E:\\mars-main\\cfgops\\mar20\\splits\\v5\\train.txt",
        "validation": "E:\\mars-main\\cfgops\\mar20\\splits\\v5\\validation.txt",
        "test": "E:\\mars-main\\cfgops\\mar20\\splits\\v5\\test.txt",
        "small": "E:\\mars-main\\cfgops\\mar20\\splits\\v5\\small.txt",
    }

    if "full" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 200  # 增加到200轮
        mcfg.backboneFreezeEpochs = [x for x in range(0, 100)]  # 前100轮冻结backbone
        mcfg.lossWeights = (8.0, 1.5, 1.5)  # 保持当前最佳loss权重
        mcfg.baseLearningRate = 5e-3  # 保持当前学习率

    if "teacher" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 10 # Modified for quick experiment
        mcfg.backboneFreezeEpochs = [x for x in range(0, 5)]
        mcfg.trainSelectedClasses = ["A{}".format(x) for x in range(1, 11)] # DO NOT MODIFY
        mcfg.checkpointModelFile = None  # 确保不使用 full 的权重
        mcfg.pretrainedBackboneUrl = None  # 确保不使用预训练权重

    if "distillation" in tags:
        mcfg.modelName = "distillation"
        # 修改教师模型文件路径
        mcfg.checkpointModelFile = "marsdata\\test_teacher\\c1.nano.teacher\\__cache__\\best_weights.pth"
        mcfg.teacherModelFile = "marsdata\\test_teacher\\c1.nano.teacher\\__cache__\\best_weights.pth"
        mcfg.distilLossWeights = (1.0, 0.05, 0.001)
        mcfg.maxEpoch = 100
        mcfg.backboneFreezeEpochs = [x for x in range(0, 25)]
        mcfg.epochValidation = False # DO NOT MODIFY
        mcfg.trainSplitName = "small" # DO NOT MODIFY
        mcfg.teacherClassIndexes = [x for x in range(0, 10)] # DO NOT MODIFY

    return mcfg
