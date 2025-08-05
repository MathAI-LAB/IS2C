# IS2C
## Paper

<b> Probability-Polarized Optimal Transport for Unsupervised Domain Adaptation </b> <br/>
![alt text](./image_PPOT.png) <br/>
Cheng-Jun Guo, [Chuan-Xian Ren*](https://scholar.google.com/citations?user=nWsPNkQAAAAJ&hl=zh-CN), [You-Wei Luo](https://scholar.google.com/citations?user=n9xRWGsAAAAJ&hl=zh-CN), [Hong Yan](https://scholar.google.com.hk/citations?user=oKwuCfAAAAAJ&hl=zh-CN)<br/>
This is the pytorch demo code for Partial Domain Adaptation via Importance Sampling-based Shift Correction (TIP 2025) | [paper](https://ieeexplore.ieee.org/document/11107265).<br/>

## Requirements:
python == 3.6.13 <br/>
torch == 1.4.0 <br/>
torchvision == 0.11.0 <br/>
cuda == 11.3 <br/>
POT == 0.8.2 <br/>

## Files structures:
├─utils.py <br/>
The image feature input, model architecture, classification function, etc. of the research method used in this code.<br/>

├─main.py <br/>
Run the main.py file for training models.<br/>

## Citation
If you find this paper useful in your research, please consider citing:
```bibtex
@inproceedings{wang2024probability,
  title={Probability-Polarized Optimal Transport for Unsupervised Domain Adaptation},
  author={Wang, Yan and Ren, Chuan-Xian and Zhai, Yi-Ming and Luo, You-Wei and Yan, Hong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={14},
  pages={15653--15661},
  year={2024}
}
```
