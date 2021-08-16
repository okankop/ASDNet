import os
import sys
import csv
import glob
import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from core.opts import parse_opts
from core.dataset import *
from core.optimization import *
from core.io import *
from core.models import *
from core.util import *




if __name__ == '__main__':
    #experiment Reproducibility
    torch.manual_seed(11)
    torch.cuda.manual_seed(22)
    torch.backends.cudnn.deterministic = True

    opt = parse_opts()


    ################################################################################
    ######################## AUDIO-VISUAL ENCODING STAGE ###########################
    ################################################################################
    if opt.stage == 'av_enc': 
        if not opt.postprocess: 
            # model creation
            model = TwoStreamNet(opt)
            has_cuda = torch.cuda.is_available()
            device = torch.device('cuda:'+opt.cuda_device_num if has_cuda else 'cpu')
            model = model.to(device)

            if opt.resume_path:
                model.load_state_dict(torch.load(opt.resume_path))
                print("Weigths are loaded from ", opt.resume_path)

            video_data_transforms = transforms.Compose([
                                        transforms.Resize(opt.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.2324, 0.2721, 0.3448), (0.1987, 0.2172, 0.2403))
                                        ])

            if not opt.forward: # AV_ENC Training
                model.train()
                log, target_models = set_up_log_and_ws_out(opt, 'av_enc')

                #Optimization config
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=opt.av_enc_learning_rate)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

                video_path = os.path.join(opt.input_video_dir, 'train')
                audio_path = os.path.join(opt.input_audio_dir, 'train')

                d_train = AV_Enc_Dataset(audio_path, video_path,
                                                  opt.annotation_train, opt.clip_length,
                                                  opt.image_size, video_data_transforms,
                                                  do_video_augment=True)
                d_val = AV_Enc_Dataset(audio_path, video_path,
                                                opt.annotation_val, opt.clip_length,
                                                opt.image_size, video_data_transforms,
                                                do_video_augment=False)


                dl_train = DataLoader(d_train, batch_size=opt.av_enc_batch_size,
                                      shuffle=True, num_workers=opt.threads)
                dl_val = DataLoader(d_val, batch_size=opt.av_enc_batch_size,
                                    shuffle=True, num_workers=opt.threads)

                model = optimize_av_enc(model, dl_train, dl_val, device,
                                              criterion, optimizer, scheduler,
                                              num_epochs=opt.epochs,
                                              models_out=target_models, log=log)

            else: # AV_ENC Feature Extraction
                model.eval()
                target_directory = 'results/av_enc_forward/'
                if not os.path.isdir(target_directory):
                    os.makedirs(target_directory)
                
                if opt.feature_extraction_subset == 'train':
                    video_path = os.path.join(opt.input_video_dir, 'train')
                    audio_path = os.path.join(opt.input_audio_dir, 'train')
                    video_list = load_train_video_set()
                    csv_file = opt.annotation_train
                elif opt.feature_extraction_subset == 'val':
                    video_path = os.path.join(opt.input_video_dir, 'train')
                    audio_path = os.path.join(opt.input_audio_dir, 'train')
                    video_list = load_val_video_set()
                    csv_file = opt.annotation_val
                elif opt.feature_extraction_subset == 'test':
                    video_path = os.path.join(opt.input_video_dir, 'test')
                    audio_path = os.path.join(opt.input_audio_dir, 'test')
                    video_list = load_test_video_set()
                    csv_file = opt.annotation_test
                else:
                    print("For AV_Enc feature extraction, select dataset subset from list: [train, val, test]")

                for video_key in video_list:
                    print('forward video ', video_key)
                    with open(target_directory+video_key+'.csv', mode='w') as vf:
                        vf_writer = csv.writer(vf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        d_val = AV_Enc_Forward_Dataset(video_key, audio_path, video_path,
                                                        csv_file, opt.clip_length,
                                                        opt.image_size, video_data_transforms,
                                                        do_video_augment=False)

                        dl_val = DataLoader(d_val, batch_size=opt.av_enc_batch_size,
                                            shuffle=False, num_workers=opt.threads)

                        for idx, dl in enumerate(dl_val):
                            print(' \t Forward iter ', idx, '/', len(dl_val), end='\r')
                            audio_data, video_data, video_id, ts, entity_id, gt = dl
                            video_data = video_data.to(device)
                            audio_data = audio_data.to(device)

                            with torch.set_grad_enabled(False):
                                preds, _, _, feats = model(audio_data, video_data)

                            feats = feats.detach().cpu().numpy()
                            for i in range(preds.size(0)):
                                vf_writer.writerow([video_id[i], ts[i], entity_id[i], float(gt[i]), float(preds[i][0]), float(preds[i][1]), list(feats[i])])

        else: # Post-processing 
            forward_dir = os.path.join(opt.save_dir, 'av_enc_forward') #Directory where you store the network predcitons
            ava_ground_truth_dir = '/usr/home/kop/ASDNet/ava_activespeaker_test_v1.0' #AVA original ground truth files
            temporary_dir = 'results/temp' #Just an empty temporary dir
            if not os.path.isdir(temporary_dir):
                os.makedirs(temporary_dir)

            # You need both to use AVA evaluation
            if not os.path.isdir('/usr/home/kop/ASDNet/final'):
                os.makedirs('/usr/home/kop/ASDNet/final')
            dataset_predictions_csv = '/usr/home/kop/ASDNet/final/AV_Enc.csv'  #file with final predictions
            dataset_gt_csv = '/usr/home/kop/ASDNet/final/gt.csv' # Utility file to use the official evaluation tool

            #cleanup temp dir
            del_files = glob.glob(temporary_dir+'/*')
            for f in del_files:
                os.remove(f)

            pred_files, gt_files = select_files(forward_dir, ava_ground_truth_dir)

            for idx, (pf, gtf) in enumerate(zip(pred_files, gt_files)):
                prediction_data = csv_to_list(pf)
                gt_data = csv_to_list(gtf)

                print('Match', os.path.basename(pf), len(prediction_data), len(gt_data))
                if len(prediction_data) != len(gt_data):
                    raise Exception('Groundtruth and prediction dont match in lenght')

                post_processed_predictions = prediction_postprocessing(prediction_data, 1)

                #reformat into ava required style
                for idx in range(len(post_processed_predictions)):
                    post_processed_predictions[idx] = [gt_data[idx][0], gt_data[idx][1],
                                                gt_data[idx][2], gt_data[idx][3],
                                                gt_data[idx][4], gt_data[idx][5],
                                                'SPEAKING_AUDIBLE', gt_data[idx][-1],
                                                '{0:.4f}'.format(post_processed_predictions[idx][-1])]

                target_csv = os.path.join(temporary_dir, os.path.basename(pf))
                write_to_file(post_processed_predictions, target_csv)

            processed_gt_files = glob.glob(temporary_dir+'/*.csv')
            processed_gt_files.sort()
            gt_files.sort()
            os.system('cat ' + ' '.join(processed_gt_files) + '> '+ dataset_predictions_csv)
            os.system('cat ' + ' '.join(gt_files) + '> '+ dataset_gt_csv)




    ################################################################################
    ##################### TEMPORAL MODELING AND ISRM STAGEs ########################
    ################################################################################
    elif opt.stage == 'tm_isrm':
        if not opt.postprocess:
            # model creation
            model = TM_ISRM_Net(num_speakers=opt.num_speakers)
            has_cuda = torch.cuda.is_available()
            device = torch.device('cuda:'+opt.cuda_device_num if has_cuda else 'cpu')
            model.to(device)

            if opt.resume_path:
                model.load_state_dict(torch.load(opt.resume_path))
                print("Weigths are loaded from ", opt.resume_path)

            if not opt.forward: # AV_ENC Training
                model.train()

                # io config
                model_name = 'TM_ISRM'+'_len_' + str(opt.seq_length) + '_stride_'+str(opt.time_stride) + '_speakers_'+str(opt.num_speakers)
                log, target_models = set_up_log_and_ws_out(opt, model_name)

                #Optimization config
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=opt.tm_isrm_learning_rate)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

                d_train = TM_ISRM_Dataset(opt.features_train_dir,
                                               seq_length=opt.seq_length,
                                               time_stride=opt.time_stride,
                                               num_speakers=opt.num_speakers)
                d_val = TM_ISRM_Dataset(opt.features_val_dir,
                                               seq_length=opt.seq_length,
                                               time_stride=opt.time_stride,
                                               num_speakers=opt.num_speakers)

                dl_train = DataLoader(d_train, batch_size=opt.tm_isrm_batch_size,
                                      shuffle=True, num_workers=opt.threads)
                dl_val = DataLoader(d_val, batch_size=opt.tm_isrm_batch_size,
                                    shuffle=False, num_workers=opt.threads)

                optimize_tm_isrm(model, dl_train, dl_val, device, criterion, optimizer,
                             scheduler, num_epochs=opt.epochs,
                             models_out=target_models, log=log)

            else: # TM_ISRM Feature Extraction
                model.eval()
                target_directory = 'results/tm_isrm_forward/'
                if not os.path.isdir(target_directory):
                    os.makedirs(target_directory)
                
                if opt.feature_extraction_subset == 'val':
                    video_list =  load_val_video_set()
                    csv_file = opt.features_val_dir
                elif opt.feature_extraction_subset == 'test':
                    video_list =  load_test_video_set()
                    csv_file = opt.features_test_dir
                else:
                    print("For TM_ISRM feature extraction, select dataset subset from list: [val, test]")

                for video_key in video_list:
                    print('forward video ', video_key)
                    with open(os.path.join(target_directory, video_key+'.csv'), mode='w') as vf:
                        vf_writer = csv.writer(vf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                        features_file = os.path.join(csv_file, video_key+'.csv')
                        d_val = TM_ISRM_Forward_Dataset(features_file, opt.seq_length, opt.time_stride, opt.num_speakers)

                        dl_val = DataLoader(d_val, batch_size=opt.tm_isrm_batch_size,
                                            shuffle=False, num_workers=opt.threads)

                        for idx, dl in enumerate(dl_val):
                            print(' \t Forward iter ', idx, '/', len(dl_val), end='\r')
                            features, video_id, ts, entity_id = dl
                            features = features.to(device)

                            with torch.set_grad_enabled(False):
                                preds = model(features)
                                bs = preds.size(0)
                                preds = preds.detach().cpu().numpy()
                            for i in range(bs):
                                vf_writer.writerow([entity_id[i], ts[i], str(preds[i][0]), str(preds[i][1])])

        else: # Post-processing 
            forward_dir = os.path.join(opt.save_dir, 'tm_isrm_forward') #Directory where you store the network predcitons
            ava_ground_truth_dir = '/usr/home/kop/ASDNet/ava_activespeaker_test_v1.0' #AVA original ground truth files
            temporary_dir = 'results/temp' #Just an empty temporary dir
            if not os.path.isdir(temporary_dir):
                os.makedirs(temporary_dir)
                
            # You need both to use AVA evaluation
            if not os.path.isdir('/usr/home/kop/ASDNet/final'):
                os.makedirs('/usr/home/kop/ASDNet/final')
            dataset_predictions_csv = '/usr/home/kop/ASDNet/final/TM_ISRM.csv'  #file with final predictions
            dataset_gt_csv = '/usr/home/kop/ASDNet/final/gt.csv' # Utility file to use the official evaluation tool

            #cleanup temp dir
            del_files = glob.glob(temporary_dir+'/*')
            for f in del_files:
                os.remove(f)

            pred_files, gt_files = select_files(forward_dir, ava_ground_truth_dir)

            for idx, pf in enumerate(pred_files):
                pred_data = csv_to_list(pf)
                gt_data = csv_to_list(os.path.join(ava_ground_truth_dir, os.path.basename(pf)[:-4]+'-activespeaker.csv'))

                print(idx, os.path.basename(pf), len(pred_data), len(gt_data))
                post_processed_data = softmax_feats(pf)

                for idx in range(len(post_processed_data)):
                    post_processed_data[idx] = [gt_data[idx][0], gt_data[idx][1],
                                                gt_data[idx][2], gt_data[idx][3],
                                                gt_data[idx][4], gt_data[idx][5],
                                                'SPEAKING_AUDIBLE', gt_data[idx][-1],
                                                '{0:.4f}'.format(post_processed_data[idx][-1])]

                target_csv = os.path.join(temporary_dir, os.path.basename(pf))
                write_to_file(post_processed_data, target_csv)

            processed_gt_files = glob.glob(temporary_dir+'/*.csv')
            processed_gt_files.sort()
            gt_files.sort()
            os.system('cat ' + ' '.join(processed_gt_files) + '> '+ dataset_predictions_csv)
            os.system('cat ' + ' '.join(gt_files) + '> '+ dataset_gt_csv)
