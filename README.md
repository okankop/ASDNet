# ASDNet

Pytorch implementation of the article [How to Design a Three-Stage Architecture for Audio-Visual Active Speaker Detection in the Wild](https://arxiv.org/pdf/2106.03932.pdf) 

<p
   align="center">
  <img src="https://github.com/okankop/ASDNet/blob/main/visuals/AV-ASD-Pipeline.jpg" align="middle" width="400" title="ASDNet pipeline" />
  <figcaption><b>Figure 1.</b>Audio-visual active speaker detection pipeline. The task is to determine if the reference speaker at frame <i>t</i> is <i>speaking</i> or <i>not-speaking</i>. The pipeline starts with audio-visual encoding of each speaker in the clip. Secondly, inter-speaker relation modeling is applied within each frame. Finally, temporal modeling is used to capture long-term relationships in natural conversations. Examples are from AVA-ActiveSpeaker.</figcaption>
</p>


## Requirements

* To create conda environment and install required libraries, please run `./scripts/dev_env.sh`.

## Dataset Preparation

* **Run `./scripts/dowloads.sh`** in order to download 3 utility files, which is necessary to preprocess AVA-ActiveSpeaker dataset.

1. **Download AVA videos** from https://github.com/cvdfoundation/ava-dataset.

2. **Extract the audio tracks** from every video in the dataset. Go to ./data/extract_audio_tracks.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos) and `target_audios` (empty directory where the audio tracks will be stored) to your local file system.

3. Slice the audio tracks by timestamp. Go to ./data/slice_audio_tracks.py in  __main__ adapt the `ava_audio_dir` (the directory with the audio tracks you extracted on step 1), `output_dir` (empty directory where you will store the sliced audio files) and  `csv` (the utility file you download previously, use the set accordingly) to your local file system.

4. Extract the face crops by timestamp. Go to ./data/extract_face_crops_time.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos), `csv_file` (the utility file you download previously, use the train/val/test set accordingly) and  `output_dir` (empty directory where you will store the face crops) to your local file system. This process will result in about 124GB extra data.

The full audio tracks obtained on step 2. will not be used anymore.


## Audio-Visual Encoding (AV_Enc): Training, Feature Extraction, Postprocessing and Results

### Training
Audio-visual encoders can be trained with the following command:
```bash
python main.py --stage av_enc \
	--audio_backbone sincdsnet \
	--video_backbone resnext101 \
	--video_backbone_pretrained_path /usr/home/kop/ASDNet/weights/kinetics_resnext_101_RGB_16_best.pth \
	--epochs 70 \
	--step_size 30 \
	--av_enc_learning_rate 3e-4 \
	--av_enc_batch_size 24 \
```

### Feature Extraction
Use `--forward` to enable feature extraction, and use `--resume_path` to specify which saved model path to use at feature extraction.

### Postprocesssing
Use `--postprocessing` to enable postprocessing, which produces `final/AV_Enc.csv` and `final/gt.csv`. 

### Getting Results
Use following command to get AV_Enc results:
```bash
python get_ava_active_speaker_performance.py -p final/AV_Enc.csv -g final/gt.csv
```



## Temporal Modeling and Inter-Speaker Relation Modeling (TM_ISRM): Training, Feature Extraction and Postprocessing

### Training
TM and ISRM stages can be trained with the following command:
```bash
python main.py --stage tm_isrm \
	--epochs 10 \
	--step_size 5 \
	--av_enc_learning_rate 3e-6 \
	--av_enc_batch_size 256 \
```

For validation results, there is no need to extract features and apply postprocessing. Training script directly produces mAP results for active speaking class. However, the following feature extraction, postprocessing scripts would be usefull for test set.

### Feature Extraction
Use `--forward` to enable feature extraction, and use `--resume_path` to specify which saved model path to use at feature extraction.

### Postprocesssing
Use `--postprocessing` to enable postprocessing, which produces `final/TM_ISRM.csv` and `final/gt.csv`. 


## Citation
If you use this code or pre-trained models, please cite the following:

```bibtex
@article{kopuklu2021asdnet,
  title={How to Design a Three-Stage Architecture for Audio-Visual Active Speaker Detection in the Wild},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Taseska, Maja and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:2106.03932},
  year={2021}
}
```


### Acknowledgements
We thank [Juan Carlos Leon Alcazar](https://github.com/fuankarion) for releasing [active-speakers-context](https://github.com/fuankarion/active-speakers-context) codebase, from which we use dataset preprocessing and data loaders. 