import argparse

# rawAudioPath = 'data/fake_train_data'
# rawLabelsPath= 'data/fake_train_labels'
# metadataPathTrain = 'data/transcription/train_small.csv'
# metadataPathTest = 'data/transcription/test_small.csv'

# rawAudioPath = '../drive/MyDrive/train_small'
# rawLabelsPath= '../drive/MyDrive/train_small_labels'
# rawAudioPath = '../musicnet/musicnet/train_data'
# rawLabelsPath= '../musicnet/musicnet/train_labels'

# guitar set
rawAudioPath = '../guitarset/audio_mono-pickup_mix'
rawLabelsPath= '../guitarset/annotation'

# rawAudioPath = '../musicnet/train_small'
# rawLabelsPath= '../musicnet/train_small_labels'
# metadataPathTrain = 'data/small_transcription/train_small.csv'
# metadataPathTest = 'data/small_transcription/test_small.csv'
metadataPathTrain = 'data/guitar_transcription/guitar_train.csv'
metadataPathTest = 'data/guitar_transcription/guitar_test.csv'



def getDefaultConfig():
    parser = setDefaultConfig(argparse.ArgumentParser())
    return parser.parse_args([])


def setDefaultConfig(parser):
    # Run parameters
    group = parser.add_argument_group('Architecture configuration',
                                      description="The arguments defining the "
                                      "model's architecture.")
    group.add_argument('--hiddenEncoder', type=int, default=512,
                       help='Hidden dimension of the encoder network.')
    group.add_argument('--hiddenGar', type=int, default=256,
                       help='Hidden dimension of the auto-regressive network')
    group.add_argument('--nPredicts', type=int, default=12,
                       help='Number of steps to predict.')
    group.add_argument('--negativeSamplingExt', type=int, default=128,
                       help='Number of negative samples to take.')
    group.add_argument('--learningRate', type=float, default=2e-4)
    group.add_argument('--schedulerStep', type=int, default=-1,
                       help='Step of the learning rate scheduler: at each '
                       'step the learning rate is divided by 2. Default: '
                       'no scheduler.')
    group.add_argument('--schedulerRamp', type=int, default=None,
                       help='Enable a warm up phase for the learning rate: '
                       'adds a linear ramp of the given size.')
    group.add_argument('--beta1', type=float, default=0.9,
                       help='Value of beta1 for the Adam optimizer')
    group.add_argument('--beta2', type=float, default=0.999,
                       help='Value of beta2 for the Adam optimizer')
    group.add_argument('--epsilon', type=float, default=1e-08,
                       help='Value of epsilon for the Adam optimizer')
    group.add_argument('--sizeWindow', type=int, default=20480,
                       help='Number of frames to consider at each batch.')
    group.add_argument('--batchSize', type=int, default=8,
                       help='Number of samples per mini batch.')
    group.add_argument('--nEpoch', type=int, default=30,
                       help='Number of epoch to run')
    group.add_argument('--samplingType', type=str, default='uniform',
                       choices=['samecategory', 'uniform',
                                'samesequence', 'sequential'],
                       help='How to sample the negative examples in the '
                       'CPC loss.')
    group.add_argument('--nLevelsPhone', type=int, default=1,
                       help='(Supervised mode only). Number of layers in '
                       'the phone classification network.')
    group.add_argument('--cpcMode', type=str, default=None,
                       choices=['reverse', 'none'],
                       help='Some variations on CPC.')
    group.add_argument('--encoderType', type=str,
                       choices=['sinc', None],
                       default='sinc',
                       help='Use SincNet or simple 1D convolution as encoder.')
    group.add_argument('--normMode', type=str, default='layerNorm',
                       choices=['instanceNorm', 'ID', 'layerNorm',
                                'batchNorm'],
                       help="Type of normalization to use in the encoder "
                       "network (default is layerNorm).")
    group.add_argument('--onEncoder', action='store_true',
                       help="(Supervised mode only) Perform the "
                       "classification on the encoder's output.")
    group.add_argument('--randomSeed', type=int, default=None,
                       help="Set a specific random seed.")
    group.add_argument('--arMode', default='transformer',
                       choices=['GRU', 'LSTM', 'RNN', 'transformer'],
                       help="Architecture to use for the auto-regressive "
                       "network (default is transformer).")
    group.add_argument('--nLevelsGRU', type=int, default=1,
                       help='Number of layers in the autoregressive network.')
    # group.add_argument('--rnnMode', type=str, default='transformer',
    #                    choices=['transformer', 'RNN', 'LSTM', 'linear',
    #                             'ffd', 'conv4', 'conv8', 'conv12'],
    #                    help="Architecture to use for the prediction network")
    group.add_argument('--dropout', action='store_true',
                       help="Add a dropout layer at the output of the "
                       "prediction network.")
    group.add_argument('--abspos', action='store_true',
                       help='If the prediction network is a transformer, '
                       'active to use absolute coordinates.')
    return parser
