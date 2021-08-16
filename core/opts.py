import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    # Main Arguments
    parser.add_argument('--stage', default='av_enc', type=str, help='(av_enc | tm_isrm')
    parser.add_argument('--forward', action='store_true', help='If true, starts extracting features')
    parser.set_defaults(forward=False)
    parser.add_argument('--postprocess', action='store_true', help='If true, concatenates all csv files and creates one')
    parser.set_defaults(postprocess=False)
    parser.add_argument('--save_dir', default='results', type=str, help='models save directory')
    parser.add_argument('--resume_path', default=None, type=str, help='Model path (.pth) to resume training or extract features')
    parser.add_argument('--optimizer', default='adam', type=str, help='Applied optimizer')
    parser.add_argument('--criterion', default='crossentropy', type=str, help='Applied loss')
    parser.add_argument('--threads', default=8, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--epochs',  default=10, type=int, help='Number of epochs to train')
    parser.add_argument('--step_size', default=5, type=int, help='Steps to reduce learning rate by gamma')
    parser.add_argument('--gamma', default=0.1, type=float, help='Ratio to drop learning rate at step size')
    parser.add_argument('--image_size', default=(160, 160), type=int, help='Used image resolution')
    parser.add_argument('--clip_length', default=16, type=int, help='Used clip length')
    parser.add_argument('--cuda_device_num', default='0', type=str, help='Used cuda device number')
    parser.add_argument('--feature_extraction_subset', default='val', type=str, help='Subset to extract features. (train | val | test)')

    # AV_ENC Arguments
    parser.add_argument('--av_enc_batch_size', default=24, type=int, help='batch size to train av_enc stage')
    parser.add_argument('--av_enc_learning_rate', default=3e-4, type=float, help='learning rate to train av_enc stage')
    parser.add_argument('--video_backbone', default='resnext101', type=str, help='(resnet18 | resnext101 | ... ')
    parser.add_argument('--video_backbone_pretrained_path', default='/usr/home/kop/ASDNet/weights/kinetics_resnext_101_RGB_16_best.pth', type=str, help='Pretrained video backbone weight')
    parser.add_argument('--audio_backbone', default='sincdsnet', type=str, help='Currently only sincdsnet is supported')
    parser.add_argument('--audio_backbone_pretrained_path', default=None, type=str, help='Pretrained audio backbone weight')
    parser.add_argument('--annotation_train', default='/usr/home/kop/ASDNet/ava_activespeaker_train_augmented.csv', type=str, help='Train annotation path')
    parser.add_argument('--annotation_val', default='/usr/home/kop/ASDNet/ava_activespeaker_val_augmented.csv', type=str, help='Val annotation path')
    parser.add_argument('--annotation_test', default='/usr/home/kop/ASDNet/ava_activespeaker_test_augmented.csv', type=str, help='Test annotation path')
    parser.add_argument('--input_audio_dir', default='/usr/home/kop/datasets/AVA_ActiveSpeaker/instance_wavs_time_v2', type=str, help='Input audio directory')
    parser.add_argument('--input_video_dir', default='/usr/home/kop/datasets/AVA_ActiveSpeaker/instance_crops', type=str, help='Input video directory')

    # TM_ISRM Arguments
    parser.add_argument('--num_speakers', default=4, type=int, help='The number of speakers including reference speaker')
    parser.add_argument('--seq_length', default=64, type=int, help='Sequence length to train the TM_ISRM stages')
    parser.add_argument('--time_stride', default=1, type=int, help='Applied stride to select features from seq_length')
    parser.add_argument('--tm_isrm_batch_size', default=256, type=int, help='batch size to train av_enc stage')
    parser.add_argument('--tm_isrm_learning_rate', default=3e-6, type=float, help='learning rate to train tm_isrm stage')
    parser.add_argument('--features_train_dir', default='results/features_resnext101_16f/train_forward', type=str, help='Features train directory')
    parser.add_argument('--features_val_dir', default='results/features_resnext101_16f/val_forward', type=str, help='Features val directory')
    parser.add_argument('--features_test_dir', default='results/features_resnext101_16f/test_forward', type=str, help='Features val directory')
    parser.add_argument('--features_train_forward_dir', default='results/tm_isrm_forward/train_forward', type=str, help='Features train forward directory')
    parser.add_argument('--features_val_forward_dir', default='results/tm_isrm_forward/val_forward', type=str, help='Features val forward directory')

    args = parser.parse_args()

    return args
