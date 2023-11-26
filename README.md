# Enhancing Personality Recognition in Dialogue

This is the official github repository for the paper:"Enhancing Personality Recognition in Dialogue by Data Augmentation and Heterogeneous Conversational Graph Networks" by Yahui Fu, Haiyue Song, Tianyu Zhao, and Tatsuya Kawahara.

Paper link will be put here after the paper is accepted. Pre-processed corpora and trained model (now model.pt is empty) will be released after we got permission.

<!-- ## Introduction

Our work focuses on improving personality recognition in dialogues, a critical aspect for enhancing human-robot interactions. The challenges addressed include the limited number of speakers in dialogue corpora and the complex modeling of interdependencies in conversations. -->

<!-- ## Key Contributions:

1. **Data Augmentation for Personality Recognition:** We propose a novel data interpolation method for speaker data augmentation to increase speaker diversity.
2. **Heterogeneous Conversational Graph Network (HC-GNN):** A new approach to model both contextual influences and inherent personality traits independently. -->

## Step1. Dependencies

Install python3, make virtual enviroment (recommended), and install python packages by:

`pip install --upgrade pip && pip -install -r requirements.txt `

## Step2. Data Preprocessing

* Big-Five label preparing, this is to convert personality questinnares to big5 labels.
  `python data_preprocessing/big5_preprocessing.py`
* Speaker-independently corpus splitting for monologue experiments
* Speaker-independently corpus splitting for dialogue experiments

## Step3. Training

1. This allows to train a MLP model on the original monologue dataset (without data augmentation).

`	python train.py`

## Folders

- `data/` contains the pre-processed corpora (now only sample data because we are waiting for permission)
- `log/` contains the log file where results are saved
- `model/` contains the trained model (now model.pt is empty, we may consider to release trained models after we got permission)

## Sample Results

This contains the best result we obtained in the paper.

`log/monologue_split_DA500k_MLP.log`

Here are some other results we obtained in the paper.

- log/monologue_split_original_MLP.log
- log/dialogue_split_original_MLP.log
- log/dialogue_split_original_HCGCN.log

## Citation

We will put the citation here once the paper is accepted.

<!-- 
`
If you find our work useful in your research, please consider citing:
@inproceedings{fu2024enhancing,
title={Enhancing Personality Recognition in Dialogue by Data Augmentation and Heterogeneous Conversational Graph Networks},
author={Fu, Yahui and Song, Haiyue and Zhao, Tianyu and Kawahara, Tatsuya},
year={2023}
}
`
-->

## Contact

For any queries related to the paper or the implementation, feel free to contact:

- Haiyue Song is in charge of the data augmentation part. [song@nlp.ist.i.kyoto-u.ac.jp](mailto:song@nlp.ist.i.kyoto-u.ac.jp)
- Yahui Fu is in charge of the HC-GNN model part. [fu.yahui.64p@st.kyoto-u.ac.jp](mailto:fu.yahui.64p@st.kyoto-u.ac.jp)
