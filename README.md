# Hypernetwork-based Physics-Driven Personalized Federated Learning for CT Imaging

This repository is a PyTorch implementation of CO3Net (accepted by IEEE Transactions on Neural Networks and Learning Systems). The Preprint Version can be downloaded at [here](https://ieeexplore.ieee.org/document/10361833).

#### Abstract
In clinical practice, computed tomography (CT) is an important noninvasive inspection technology to provide patients' anatomical information. However, its potential radiation risk is an unavoidable problem which raises people's concerns. Recently, deep learning-based methods have achieved promising results in CT reconstruction, but these methods usually require the centralized collection of large amounts of data for training from specific scanning protocols, which leads to serious domain shift and privacy concerns. To relieve these problems, in this paper, we propose a hypernetwork-based physics-driven personalized federated learning method (HyperFed) for CT imaging. The basic assumption of the proposed HyperFed is that the optimization problem for each domain can be divided into two subproblems: local data adaption and global CT imaging problems, which are implemented by an institution-specific physics-driven hypernetwork and a global-sharing imaging network, respectively. Learning stable and effective invariant features from different data distributions is the main purpose of global-sharing imaging network. Inspired by the physical process of CT imaging, we carefully design physics-driven hypernetwork for each domain to obtain hyperparameters from specific physical scanning protocol to condition the global-sharing imaging network, so that we can achieve personalized local CT reconstruction. Experiments show that HyperFed achieves competitive performance in the comparison with several other state-of-the-art methods. It is believed as a promising direction to improve CT imaging quality and personalize the needs of different institutions or scanners without data sharing.


#### Citation
If our work is valuable to you, please cite our work:
```
@article{yang2023hyperfed,
  title={Hypernetwork-based Physics-Driven Personalized Federated Learning for CT Imaging},
  author={Yang, Ziyuan and Xia, Wenjun and Lu, Zexin and Chen, Yingyu and Li, Xiaoxiao and Zhang, Yi},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```


#### Requirements

Our codes were implemented by ```PyTorch 1.10``` and ```11.3``` CUDA version. If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```

Our implementation is based on [CTLib](https://github.com/xiawj-hub/CTLIB) in simulating data and training IR-based methods. If you have an interest in data simulation and IR-based networks, we recommend installing it. Furthermore, HyperFed can be easily integrated into transformer-based methods with minor modifications.

#### Acknowledgments
Special thanks for Dr. Xia, Dr. Li and Dr. Zhang!

#### Contact
If you have any question or suggestion to our work, please feel free to contact me. My email is cziyuanyang@gmail.com.

