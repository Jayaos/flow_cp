## Flow-based Conformal Prediction for Multi-dimensional Time Series

This repository contains soruce code of the method in the paper **["Flow-based Conformal Prediction for Multi-dimensional Time Series"](https://arxiv.org/pdf/2502.05709)**.


<p align="center">
  <img src="images/fcp_overall_method.png" width="400"/>
</p>


### Implementation Example

The code was written in Python 3.9.13 with torch 2.2.2+cu118. Other dependencies are available in requirements.txt.

We provide a colab notebooks to go through obtaining the results of base predictor and conducting experiments using FCP.


#### Obtaining results of base predictor

In this colab notebook, we train base predictors and obtain results on wind data. GPU is not strictly required.

<a target="_blank" href="https://colab.research.google.com/github/Jayaos/flow_test/blob/master/base_predictor_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


#### Implementing FCP

In this colab notebook, we can reproduce experiments using FCP on wind 2d data. GPU is required.

<a target="_blank" href="https://colab.research.google.com/github/Jayaos/flow_test/blob/master/fcp_implementation_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>




### Citation

```bibtex
@article{lee2026flow,
  title={Flow-based Conformal Prediction for Multi-dimensional Time Series},
  author={Lee, Junghwan and Xu, Chen and Xie, Yao},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}

