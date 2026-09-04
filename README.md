# Cond-UNet: Cross-Domain Ultrasound Segmentation

This repository contains the official PyTorch implementation for the BMVC 2026 paper: [A New Multicenter Testicular US Dataset and a Lightweight Cond-UNet for Generalization in US Segmentation](https://federicobolelli.it/media/publications/pdfs/0475.pdf).

Here you will find the code to run our model in inference and to fine-tune it on your own ultrasound image dataset.

Before running the code, copy `utils/paths.example.py` to `utils/paths.py` and set the local dataset and checkpoint paths.

<p align="center">
  <img src="film_unet.png" alt="IM-Fuse overview" width="">
  <br>
  <em>
    Overview of the proposed Cond-UNet architecture.
  </em>
</p>


## Dataset

TesticulUS is a multi-institutional testicular ultrasound dataset collected at **two Italian clinical centers**, capturing variability in acquisition protocols and devices for cross-domain evaluation.

Each image is paired with a **pixel-wise expert-annotated segmentation mask**. The first 860 images additionally provide labels for **parenchymal inhomogeneity** classification. The dataset supports:

- Cross-domain generalization experiments  
- Fine-tuning and transfer learning  
- Robustness benchmarking  

**Summary**
- Images: 1,054
- Institutions: 2 Italian clinical centers
- Modality: Testicular ultrasound
- Annotations: Expert pixel-wise segmentation masks for every image
- Classification: Parenchymal-inhomogeneity labels for the first 860 images
- License: CC BY-NC-SA 4.0

The dataset card is available on [Hugging Face](https://huggingface.co/datasets/AImageLab-Zip/TesticulUS). Download access is provided through the [Ditto platform](https://ditto.ing.unimore.it/testiculus/).

## Inference with our Pre-trained Model

Pre-trained Cond-UNet Attention weights and a Transformers-compatible image-segmentation pipeline are available on [Hugging Face](https://huggingface.co/AImageLab-Zip/US_Cond-UNet).

```python
from transformers import pipeline

segmenter = pipeline(
    "image-segmentation",
    model="AImageLab-Zip/US_Cond-UNet",
    trust_remote_code=True,
)
result = segmenter("ultrasound.png", organ_id=4)
```

See the model card for the organ-ID mapping and additional usage details.

## Citation

If you use the model or dataset for segmentation, please cite:

```bibtex
@inproceedings{morelli2026new,
  title={A New Multicenter Testicular US Dataset and a Lightweight Cond-UNet for Generalization in US Segmentation},
  author={Morelli, Nicola and Marchesini, Kevin and Santi, Daniele and Grana, Costantino and Bolelli, Federico and others},
  booktitle={Proceedings of the British Machine Vision Conference},
  year={2026}
}
```
