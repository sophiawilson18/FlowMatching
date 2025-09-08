# Flow Matching for Spatiotemporal Forecasting in Data Space

This repository contains experiments exploring FLow Matching for spatiotempoeral forecasting in data space. The data used in these experiments is downloaded from the [The Well dataset](https://github.com/PolymathicAI/the_well/tree/master). The repository contains a small dataset to get you started. 

## Acknowledgements and Original Sources

The code in this repository are adapted from:

- [Elucidating the Design Choice of Probability Paths in Flow Matching for Forecasting]([https://arxiv.org/abs/2410.03229](https://arxiv.org/html/2410.03229v1#bib.bib14)) by Lim et al. (2025), licensed under CC BY 4.0.

We have modified the original code to fit the scope of this project.

### 1. Clone this repository
```
git clone https://github.com/sophiawilson18/FlowMatching.git
cd FlowMatching
```

### 2. Set up the conda environment
We recommend using conda for managing dependencies:

```
conda create -n flowmatching python=3.10
conda activate flowmatching
pip install -r requirements.txt
```

### 3. Running Experiments

The original implementation of Flow Matching was performed in latent space, using an autoencoder that could be trained either separately or jointly (end-to-end) with the Flow Matching model. In contrast, our implementation focuses on performing Flow Matching directly in data space.


To run Flow Matching in data space, use:
```
carbontracker python train_fm.py --run-name fm_dataspace \
   --scale 16 --train_option data-space --epochs 1 \
```

### License

This repository itself is licensed under the MIT License.
Parts of the code adapted from Lim et al. (2025) are licensed under CC BY 4.0.

### Contact

For questions or collaborations, feel free to open an issue or contact me via email: sophia.wilson@di.ku.dk


