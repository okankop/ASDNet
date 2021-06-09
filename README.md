# ASDNet

Pytorch implementation of the article [How to Design a Three-Stage Architecture for Audio-Visual Active Speaker Detection in the Wild](https://arxiv.org/pdf/2106.03932.pdf) 

<p
   align="center">
  <img src="https://github.com/okankop/ASDNet/blob/main/visuals/AV-ASD-Pipeline.jpg" align="middle" width="400" title="ASDNet pipeline" />
  <figcaption><b>Figure 1.</b>Audio-visual active speaker detection pipeline. The task is to determine if the reference speaker at frame <i>t</i> is <i>speaking</i> or <i>not-speaking</i>. The pipeline starts with audio-visual encoding of each speaker in the clip. Secondly, inter-speaker relation modeling is applied within each frame. Finally, temporal modeling is used to capture long-term relationships in natural conversations. Examples are from AVA-ActiveSpeaker.</figcaption>
</p>


### The code will be uploaded soon!


### Citation
If you use this code or pre-trained models, please cite the following:

```bibtex
@article{kopuklu2021asdnet,
  title={How to Design a Three-Stage Architecture for Audio-Visual Active Speaker Detection in the Wild},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Taseska, Maja and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:2106.03932},
  year={2021}
}
```
