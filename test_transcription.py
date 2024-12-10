try:
    # noinspection PyUnresolvedReferences
    import comet_ml
except ImportError:
    pass
import torch
from dataloader import AudioBatchData
from model import CPCEncoder, CPCModel, CPCUnsupersivedCriterion, loadModel, getAR, CategoryCriterion, TranscriptionCriterion
from trainer import testModel
from datetime import datetime
import os
import argparse
from default_config import setDefaultConfig, rawAudioPath, metadataPathTrain, metadataPathTest
import sys
import random
from utils import setSeed, getCheckpointData, loadArgs, SchedulerCombiner, rampSchedulingFunction
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from main import loadCriterion

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')

    # Default arguments:
    parser = setDefaultConfig(parser)

    groupDb = parser.add_argument_group('Dataset')
    groupDb.add_argument('--ignoreCache', action='store_true',
                         help='Activate if the dataset has been modified '
                              'since the last training session.')
    groupDb.add_argument('--chunkSize', type=int, default=5e8,
                         help='Size (in bytes) of a data chunk')
    groupDb.add_argument('--maxChunksInMem', type=int, default=2,
                         help='Maximal amount of data chunks a dataset '
                              'can hold in memory at any given time')
    groupDb.add_argument('--labelsBy', type=str, default='id',
                         help="What attribute of the data set to use as labels. Only important if 'samplingType' "
                              "is 'samecategory'")

    group_supervised = parser.add_argument_group('Supervised mode')
    group_supervised.add_argument('--supervised', action='store_true',
                                  help='Disable the CPC loss and activate '
                                  'the supervised mode. By default, the supervised '
                                  'training method is the ensemble classification.')
    group_supervised.add_argument('--task', type=str, default='transcription',
                                   help='Type of the donwstream task if in supeprvised mode. '
                                        'Currently supported tasks are classification and transcription.')
    group_supervised.add_argument('--transcriptionWindow', type=int,
                                   help='Size of the transcription window (in ms) in the transcription downstream task.')

    groupSave = parser.add_argument_group('Save')
    groupSave.add_argument('--pathCheckpoint', type=str, default="./checkpoints",
                           help="Path of the output directory.")
    groupSave.add_argument('--loggingStep', type=int, default=1000)
    groupSave.add_argument('--saveStep', type=int, default=1,
                           help="Frequency (in epochs) at which a checkpoint "
                                "should be saved")
    groupSave.add_argument('--log2Board', type=int, default=0,
                           help="Defines what (if any) data to log to Comet.ml:\n"
                                "\t0 : do not log to Comet\n\t1 : log losses and accuracy\n\t>1 : log histograms of "
                                "weights and gradients.\nFor log2Board > 0 you will need to provide Comet.ml "
                                "credentials.")

    groupLoad = parser.add_argument_group('Load')
    groupLoad.add_argument('--load', type=str, default=None, nargs='*',
                           help="Load an existing checkpoint. Should give a path "
                                "to a .pt file. The directory containing the file to "
                                "load should also have a 'checkpoint.logs' and a "
                                "'checkpoint.args'")
    groupLoad.add_argument('--loadCriterion', action='store_true',
                           help="If --load is activated, load the state of the "
                                "training criterion as well as the state of the "
                                "feature network (encoder + AR)")

    config = parser.parse_args(argv)

    if config.load is not None:
        config.load = [os.path.abspath(x) for x in config.load]

    return config

def main(config):
    if not os.path.exists(metadataPathTest):
        print('test metadata missing')
        sys.exit()

    config = parseArgs(config)

    useGPU = torch.cuda.is_available()
    data_dir = 'data/small_transcription'
    musicNetMetadataTranscript = pd.read_csv(f'{data_dir}/metadata_transcript_test.csv')

    metadataTest = pd.read_csv(metadataPathTest).head(10)
    metadataTest = musicNetMetadataTranscript[musicNetMetadataTranscript['id'].isin(metadataTest.id)]

    metadataTest.to_csv(f'{data_dir}/test_metadata_transcription_{config.labelsBy}.csv')
    print('test shape', metadataTest.shape)
    chunk_output = 'data/small_transcription/test_chunks_small/'

    print("Loading the testing dataset")
    testDataset = AudioBatchData(rawAudioPath=rawAudioPath,
                                  metadata=metadataTest,
                                  sizeWindow=config.sizeWindow,
                                  labelsBy=config.labelsBy,
                                  outputPath=chunk_output + 'train_data/train',
                                  CHUNK_SIZE=config.chunkSize,
                                  NUM_CHUNKS_INMEM=config.maxChunksInMem,
                                  useGPU=useGPU,
                                  transcript_window=config.transcriptionWindow)
    print("Testing dataset loaded")
    print("dataset len: ", len(testDataset))

    if config.load is not None:
        cpcModel, config.hiddenGar, config.hiddenEncoder = loadModel(config.load, config)
    else:
        print("please load model")
        sys.exit()
    
    batchSize = config.batchSize
    cpcModel.supervised = config.supervised

    if config.load is not None and config.loadCriterion:
        cpcCriterion = loadCriterion(config.load[0], cpcModel.gEncoder.DOWNSAMPLING,
                                     len(metadataTest[config.labelsBy].unique()), config)
    else:
        print('provide trained criterion')
        sys.exit()

    if useGPU:
        cpcCriterion.cuda()
        cpcModel.cuda()
    
    testLoader = testDataset.getDataLoader(batchSize, 'uniform', False, numWorkers=0)
    testLogs = testModel(testLoader, cpcModel, cpcCriterion, useGPU)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
