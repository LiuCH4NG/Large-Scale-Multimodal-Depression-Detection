# Large-Scale-Multimodal-Depression-Detection
Large-Scale Multimodal Depression Detection: Integrated multi-modal data for depression detection with Multi-Modal Machine Learning Model
----

**python implementation**

<!-- ```python
Version :   0.0.1  
Author  :   Md Rezwanul Haque
Email   :   rezwan@uwaterloo.ca 
``` -->
---
### **Related resources**:

**LOCAL ENVIRONMENT**  
```python
OS          :   Ubuntu 24.04.2 LTS       
Memory      :   128.0 GiB
Processor   :   Intel® Xeon® w5-3425 × 24
Graphics    :   2 x (NVIDIA RTX A6000)
GPU Memory  :   2 x (48 GB) = 96 GB
CPU(s)      :   24
Gnome       :   46.0 
```
---

### 1. Prepare Datasets

We use the [D-Vlog](https://doi.org/10.1609/aaai.v36i11.21483) and [LMVD](https://arxiv.org/abs/2407.00024) dataset, proposed in this paper. For the D-Vlog dataset, please fill in the form at the bottom of the [dataset website](https://sites.google.com/view/jeewoo-yoon/dataset), and send a request email to the [author](mailto:yoonjeewoo@gmail.com). For the LMVD dataset, please download features on the released [Baidu Netdisk website](https://pan.baidu.com/s/1gviwLfbFcRSaARP5oT9yZQ?pwd=tvwa) or [figshare](https://figshare.com/articles/dataset/LMVD/25698351). 

Following D-Vlog's setup, the dataset is split into train, validation and test sets with a 7:1:2 ratio. For the LMVD without official splitting, we randomly split the LMVD with a 7:1:2 ratio ~~a 8:1:1 ratio~~ and the specific division is stored in `../data/lmvd-dataset/lmvd_labels.csv'. 

Furthermore, you can run ``lmvd_extract_npy.py`` to obtain .npy features to train the model. You also can make labels with this code ``lmvd_prepare_labels.py``.

### 2. [DepMamba] Training and Testing

#### Training

```bash
$ python main.py --train True --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --dataset dvlog-dataset

$ python main.py --train True --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --dataset lmvd-dataset
```

#### Testing

```bash
$ python main.py --model MultiModalDepDet --dataset dvlog-dataset

$ python main.py --model MultiModalDepDet --dataset lmvd-dataset
```


## 📖 Citation

- If you find this project useful for your research, please cite [our paper](https://arxiv.org/abs/2409.15936):

```bibtex
@inproceedings{yedepmamba,
  title={DepMamba: Progressive Fusion Mamba for Multimodal Depression Detection},
  author = {Jiaxin Ye and Junping Zhang and Hongming Shan},
  booktitle = {ICASSP 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Hyderabad, India, April 06-11, 2025},
  pages={1--5},
  year = {2025}
}
```