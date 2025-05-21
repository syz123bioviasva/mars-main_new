import os
import torch
import numpy as np
from misc.log import log
from tqdm import tqdm
from dl.vocdataset import VocDataset
from factory.modelfactory import MarsModelFactory
from train.opt import MarsOptimizerFactory
from train.sched import MarsLearningRateSchedulerFactory


class MarsBaseTrainer(object):
    def __init__(self, mcfg):
        self.mcfg = mcfg
        self.bestLoss = np.nan
        self.bestCacheFile = self.mcfg.epochBestWeightsPath()
        self.epochCacheFile = self.mcfg.epochCachePath()
        self.epochInfoFile = self.mcfg.epochInfoPath()
        self.checkpointFiles = [
            self.epochCacheFile,
            self.epochInfoFile,
        ]
        if self.mcfg.epochValidation:
            self.checkpointFiles.append(self.bestCacheFile)
        self.backboneFreezed = False

    def initTrainDataLoader(self):
        return VocDataset.getDataLoader(mcfg=self.mcfg, splitName=self.mcfg.trainSplitName, isTest=False, fullInfo=False, selectedClasses=self.mcfg.trainSelectedClasses)

    def initValidationDataLoader(self):
        if not self.mcfg.epochValidation:
            return None
        return VocDataset.getDataLoader(mcfg=self.mcfg, splitName=self.mcfg.validationSplitName, isTest=True, fullInfo=False, selectedClasses=self.mcfg.trainSelectedClasses)

    def initModel(self):
        if not self.mcfg.nobuf and all(os.path.exists(x) for x in self.checkpointFiles): # resume from checkpoint to continue training
            model = MarsModelFactory.loadPretrainedModel(self.mcfg, self.epochCacheFile)
            startEpoch = None
            with open(self.epochInfoFile) as f:
                lines = f.readlines()
                for line in lines:
                    tokens = line.split("=")
                    if len(tokens) != 2:
                        continue
                    if tokens[0] == "last_saved_epoch":
                        startEpoch = int(tokens[1])
                    if tokens[0] == "best_loss":
                        self.bestLoss = float(tokens[1])
            if startEpoch is None or (np.isnan(self.bestLoss) and self.mcfg.epochValidation):
                raise ValueError("Failed to load last epoch info from file: {}".format(self.epochInfoFile))
            if startEpoch < self.mcfg.maxEpoch:
                log.yellow("Checkpoint loaded: resuming from epoch {}".format(startEpoch))
            return model, startEpoch

        if self.mcfg.checkpointModelFile is not None: # use model from previous run, but start epoch from zero
            model = MarsModelFactory.loadPretrainedModel(self.mcfg, self.mcfg.checkpointModelFile)
            return model, 0

        model = MarsModelFactory.loadNewModel(self.mcfg, self.mcfg.pretrainedBackboneUrl)
        return model, 0

    def initLoss(self, model):
        return model.getTrainLoss()

    def initOptimizer(self, model):
        return MarsOptimizerFactory.initOptimizer(self.mcfg, model)

    def initScheduler(self, opt):
        return MarsLearningRateSchedulerFactory.initScheduler(self.mcfg, opt)

    def preEpochSetup(self, model, epoch):
        if self.mcfg.backboneFreezeEpochs is not None:
            if epoch in self.mcfg.backboneFreezeEpochs:
                model.freezeBackbone()
                self.backboneFreezed = True
            else:
                model.unfreezeBackbone()
                self.backboneFreezed = False

    def fitOneEpoch(self, model, loss, dataLoader, optimizer, epoch):
        trainLoss = 0
        model.setInferenceMode(False)
        numBatches = int(len(dataLoader.dataset) / dataLoader.batch_size)
        progressBar = tqdm(total=numBatches, desc="Epoch {}/{}".format(epoch + 1, self.mcfg.maxEpoch), postfix=dict, mininterval=0.5, ascii=False, ncols=130)

        for batchIndex, batch in enumerate(dataLoader):
            images, labels = batch
            images = images.to(self.mcfg.device)
            labels = labels.to(self.mcfg.device)
            optimizer.zero_grad()

            output = model(images)
            stepLoss = loss(output, labels)
            stepLoss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            trainLoss += stepLoss.item()
            progressBar.set_postfix(trainLossPerBatch=trainLoss / (batchIndex + 1), backboneFreezed=self.backboneFreezed)
            progressBar.update(1)

        progressBar.close()
        return trainLoss

    def epochValidation(self, model, loss, dataLoader, epoch):
        if not self.mcfg.epochValidation:
            return np.nan

        validationLoss = 0
        model.setInferenceMode(True)
        numBatches = int(len(dataLoader.dataset) / dataLoader.batch_size)
        progressBar = tqdm(total=numBatches, desc="Epoch {}/{}".format(epoch + 1, self.mcfg.maxEpoch), postfix=dict, mininterval=0.5, ascii=False, ncols=100)

        for batchIndex, batch in enumerate(dataLoader):
            images, labels = batch
            images = images.to(self.mcfg.device)
            labels = labels.to(self.mcfg.device)

            output = model(images)
            stepLoss = loss(output, labels)

            validationLoss += stepLoss.item()
            progressBar.set_postfix(validationLossPerBatch=validationLoss / (batchIndex + 1))
            progressBar.update(1)

        progressBar.close()
        return validationLoss

    def run(self):
        log.cyan("Mars trainer running...")

        model, startEpoch = self.initModel()
        if startEpoch >= self.mcfg.maxEpoch:
            log.inf("Training skipped")
            return

        loss = self.initLoss(model)
        opt = self.initOptimizer(model)
        scheduler = self.initScheduler(opt)
        trainLoader = self.initTrainDataLoader()
        validationLoader = self.initValidationDataLoader()

        for epoch in range(startEpoch, self.mcfg.maxEpoch):
            self.preEpochSetup(model, epoch)
            scheduler.updateLearningRate(epoch)
            trainLoss = self.fitOneEpoch(
                model=model,
                loss=loss,
                dataLoader=trainLoader,
                optimizer=opt,
                epoch=epoch,
            )
            validationLoss = self.epochValidation(
                model=model,
                loss=loss,
                dataLoader=validationLoader,
                epoch=epoch,
            )
            self.epochSave(epoch, model, trainLoss, validationLoss)

            # Print epoch losses after saving
            log.inf("Epoch {}/{}: Train Loss = {:.3f}, Validation Loss = {:.3f}".format(epoch + 1, self.mcfg.maxEpoch, trainLoss / len(trainLoader), validationLoss / len(validationLoader) if validationLoader is not None else float('nan')))

        log.inf("Mars trainer finished with max epoch at {}".format(self.mcfg.maxEpoch))

    def epochSave(self, epoch, model, trainLoss, validationLoss):
        model.save(self.epochCacheFile)
        if self.mcfg.epochValidation and (np.isnan(self.bestLoss) or validationLoss < self.bestLoss):
            log.green("Caching best weights at epoch {}...".format(epoch + 1))
            model.save(self.bestCacheFile)
            self.bestLoss = validationLoss
        with open(self.epochInfoFile, "w") as f:
            f.write("last_saved_epoch={}\n".format(epoch + 1))
            f.write("train_loss={}\n".format(trainLoss))
            f.write("validation_loss={}\n".format(validationLoss))
            f.write("best_loss_epoch={}\n".format(epoch + 1))
            f.write("best_loss={}\n".format(self.bestLoss))


def getTrainer(mcfg):
    return MarsBaseTrainer(mcfg)
