# Enhancing Personality Recognition in Dialogue

This is the official github repository for the paper:"Enhancing Personality Recognition in Dialogue by Data Augmentation and Heterogeneous Conversational Graph Networks" by Yahui Fu, Haiyue Song, Tianyu Zhao, and Tatsuya Kawahara. 
This work has been accepted to IWSDS 2024.

<!-- ## Introduction

Our work focuses on improving personality recognition in dialogues, a critical aspect for enhancing human-robot interactions. The challenges addressed include the limited number of speakers in dialogue corpora and the complex modeling of interdependencies in conversations. -->

<!-- ## Key Contributions:

1. **Data Augmentation for Personality Recognition:** We propose a novel data interpolation method for speaker data augmentation to increase speaker diversity.
2. **Heterogeneous Conversational Graph Network (HC-GNN):** A new approach to model both contextual influences and inherent personality traits independently. -->

## Folder Structure

- `data/` contains the pre-processed corpora (sample data is the placeholder)
- `log/` contains the log file where results are saved
- `model/` contains the trained model (model.pt is the placeholder)

## Step1. Dependencies Installation

Install python3, make virtual enviroment (recommended), and install python packages by:

`pip install --upgrade pip && pip -install -r requirements.txt `

## Step2. Data Preprocessing

We have already put the pre-processed corpora in `data/` folder. If you want to re-run the preprocessing by yourself, please follow the steps below:

* Big-Five label preparation, this is to convert the personality questionnaire to big5 labels.
    - `python big5_preprocessing.py`
* Speaker-independently corpus splitting for monologue experiments
    - `python nocontext_dataset_split.py`
* Speaker-independently corpus splitting for dialogue experiments
    - `python context_dataset_split.py`
* Speaker-independently monologue data augmentation
    - `python nocontext_data_augmentation.py`
* Speaker-independently dialogue data augmentation
    - `python context_data_augmentation.py`

## Step3. Training and Evaluation

1. This allows to train a MLP model on the original monologue dataset without data augmentation.
* `python train.py`

2. Here are other settings for training:

* MLP model on the augmented monologue dataset.
    - `python train.py --data_folder ./data/monologue_split_500k`
* MLP model on the original dialogue dataset.
    - `python train.py --data_folder ./data/dialogue_split_original --context 1 --context_model_type linear`
* Proposed HCGNN model on the original dialogue dataset.
    - `python train.py --data_folder ./data/dialogue_split_original --context 1 --context_model_type gcn-nospk2pred-lastnode --model_variant hcgnn`
* For more details about the arguments, please refer to `train.py --help`.

## Sample Results

This contains the best result we obtained in the paper, results on the test set are shown in the last several lines in the log file:

- log/monologue_split_500k_MLP.log

Here are some other results we obtained in the paper:

- log/monologue_split_original_MLP.log
- log/dialogue_split_original_MLP.log
- log/dialogue_split_original_HCGNN.log

## Citation

If you find our work useful in your research, please consider citing:
```
@article{fu2024enhancing, 
  title={Enhancing Personality Recognition in Dialogue by Data Augmentation and Heterogeneous Conversational Graph Networks},
  author={Fu, Yahui and Song, Haiyue and Zhao, Tianyu and Kawahara, Tatsuya},
  journal={arXiv preprint arXiv:2401.05871},
  year={2024}
}
```
## Contact

For any queries related to the paper or the implementation, feel free to contact:

- Haiyue Song is in charge of the data augmentation part. [haiyue.song@nict.go.jp](mailto:haiyue.song@nict.go.jp)
- Yahui Fu is in charge of the HC-GNN model part. [fu.yahui.64p@st.kyoto-u.ac.jp](mailto:fu.yahui.64p@st.kyoto-u.ac.jp)
