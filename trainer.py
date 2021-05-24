import numpy as np
import torch
import time
from copy import deepcopy
# import matplotlib.pyplot as plt
import json


def update_logs(logs, logStep, prevlogs=None):
    out = {}
    for key in logs:
        out[key] = deepcopy(logs[key])

        if prevlogs is not None:
            out[key] -= prevlogs[key]
        out[key] /= logStep
    return out


def save_logs(data, pathLogs):
    with open(pathLogs, 'w') as file:
        json.dump(data, file, indent=2)


def save_checkpoint(model_state, criterion_state, optimizer_state, best_state,
                    path_checkpoint):

    state_dict = {"gEncoder": model_state,
                  "cpcCriterion": criterion_state,
                  "optimizer": optimizer_state,
                  "best": best_state}

    torch.save(state_dict, path_checkpoint)


def show_logs(text, logs):
    print("")
    print('-' * 50)
    print(text)

    for key in logs:

        if key == "iter":
            continue

        nPredicts = logs[key].shape[0]

        strSteps = ['Step'] + [str(s) for s in range(1, nPredicts + 1)]
        formatCommand = ' '.join(['{:>16}' for _ in range(nPredicts + 1)])
        print(formatCommand.format(*strSteps))

        strLog = [key] + ["{:10.6f}".format(s) for s in logs[key]]
        print(formatCommand.format(*strLog))

    print('-' * 50)


def trainStep(dataLoader,
              cpcModel,
              cpcCriterion,
              optimizer,
              loggingStep,
              useGPU):
    cpcModel.train()
    cpcCriterion.train()

    startTime = time.perf_counter()
    n_examples = 0
    logs, lastlogs = {}, None
    iterCtr = 0
    for step, fulldata in enumerate(dataLoader):
        batchData, label = fulldata
        # if step % 100 == 0:
        #     print("Step: ", step)
        #     print("Batch shape: ", batchData.shape)
        #     print("Labels: ", label)
        #     plt.figure()
        #     plt.plot(batchData[0, :, :].view(-1).detach().cpu().numpy())
        #     plt.plot(batchData[1, :, :].view(-1).detach().cpu().numpy())
        #     plt.plot(batchData[2, :, :].view(-1).detach().cpu().numpy())
        #     plt.plot(batchData[3, :, :].view(-1).detach().cpu().numpy())
        #     plt.plot(batchData[4, :, :].view(-1).detach().cpu().numpy())
        #     plt.plot(batchData[5, :, :].view(-1).detach().cpu().numpy())
        #     plt.plot(batchData[6, :, :].view(-1).detach().cpu().numpy())
        #     plt.plot(batchData[7, :, :].view(-1).detach().cpu().numpy())
        #     plt.show()
        # assert False
        n_examples += batchData.size(0)
        if useGPU:
            batchData = batchData.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
        c_feature, encoded_data, label = cpcModel(batchData, label)
        allLosses, allAcc = cpcCriterion(c_feature, encoded_data)
        totLoss = allLosses.sum()

        totLoss.backward()

        # Show grads ?
        optimizer.step()
        optimizer.zero_grad()

        if "locLoss_train" not in logs:
            logs["locLoss_train"] = np.zeros(allLosses.size(1))
            logs["locAcc_train"] = np.zeros(allLosses.size(1))

        iterCtr += 1
        logs["locLoss_train"] += (allLosses.mean(dim=0)).detach().cpu().numpy()
        logs["locAcc_train"] += (allAcc.mean(dim=0)).cpu().numpy()

        if (step + 1) % loggingStep == 0:
            new_time = time.perf_counter()
            elapsed = new_time - startTime
            print(f"Update {step + 1}")
            print(f"elapsed: {elapsed:.1f} s")
            print(
                f"{1000.0 * elapsed / loggingStep:.1f} ms per batch, {1000.0 * elapsed / n_examples:.1f} ms / example")
            locLogs = update_logs(logs, loggingStep, lastlogs)
            lastlogs = deepcopy(logs)
            show_logs("Training loss", locLogs)
            startTime, n_examples = new_time, 0

    logs = update_logs(logs, iterCtr)
    logs["iter"] = iterCtr
    show_logs("Average training loss on epoch", logs)
    return logs


def valStep(dataLoader,
            cpcModel,
            cpcCriterion,
            useGPU):
    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iterCtr = 0

    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata

        if useGPU:
            batchData = batchData.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        with torch.no_grad():
            c_feature, encoded_data, label = cpcModel(batchData, label)
            allLosses, allAcc = cpcCriterion(c_feature, encoded_data)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(1))
            logs["locAcc_val"] = np.zeros(allLosses.size(1))

        iterCtr += 1
        logs["locLoss_val"] += allLosses.mean(dim=0).cpu().numpy()
        logs["locAcc_val"] += allAcc.mean(dim=0).cpu().numpy()

    logs = update_logs(logs, iterCtr)
    logs["iter"] = iterCtr
    show_logs("Validation loss:", logs)
    return logs


def run(trainDataset,
        valDataset,
        batchSize,
        samplingMode,
        cpcModel,
        cpcCriterion,
        nEpoch,
        optimizer,
        pathCheckpoint,
        logs,
        useGPU):
    print(f"Running {nEpoch} epochs")
    startEpoch = len(logs["epoch"])
    bestAcc = 0
    bestStateDict = None
    startTime = time.time()

    for epoch in range(startEpoch, nEpoch):

        print(f"Starting epoch {epoch}")
        trainLoader = trainDataset.getDataLoader(batchSize, samplingMode,
                                                 True, numWorkers=0)

        valLoader = valDataset.getDataLoader(batchSize, 'sequential', False,
                                             numWorkers=0)

        print("Training dataset %d batches, Validation dataset %d batches, batch size %d" %
              (len(trainLoader), len(valLoader), batchSize))

        locLogsTrain = trainStep(trainLoader, cpcModel, cpcCriterion, optimizer, logs["logging_step"], useGPU)

        locLogsVal = valStep(valLoader, cpcModel, cpcCriterion, useGPU)

        print(f'Ran {epoch + 1} epochs '
              f'in {time.time() - startTime:.2f} seconds')

        if useGPU:
            torch.cuda.empty_cache()

        currentAccuracy = float(locLogsVal["locAcc_val"].mean())
        if currentAccuracy > bestAcc:
            bestStateDict = cpcModel.state_dict()

        for key, value in dict(locLogsTrain, **locLogsVal).items():
            if key not in logs:
                logs[key] = [None for _ in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        logs["epoch"].append(epoch)

        if pathCheckpoint is not None and (epoch % logs["saveStep"] == 0 or epoch == nEpoch - 1):
            modelStateDict = cpcModel.state_dict()
            criterionStateDict = cpcCriterion.state_dict()

            save_checkpoint(modelStateDict, criterionStateDict, optimizer.state_dict(), bestStateDict,
                            f"{pathCheckpoint}_{epoch}.pt")
            save_logs(logs, pathCheckpoint + "_logs.json")
