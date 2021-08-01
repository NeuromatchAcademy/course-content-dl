# Ideas

Different categories of DL projects (in order of increasing expertise):
1. As an analysis toolkit to solve a problem
2. As a model of Brain or Behavior
3. Making Deep Learning pipeline more efficient / Understanding why a DL framework works (Conceptual Question). Can we make neural networks more like the brain?

# Datasets

## [NMA curated](https://github.com/NeuromatchAcademy/course-content/tree/master/projects)  
[Steinmetz Neuropixels behavior](https://github.com/NeuromatchAcademy/course-content/blob/master/projects/neurons/load_steinmetz_decisions.ipynb)  
[Steinmetz LFP behavior](https://github.com/NeuromatchAcademy/course-content/blob/master/projects/neurons/load_steinmetz_extra.ipynb)  
[Stringer spontaneous](https://github.com/NeuromatchAcademy/course-content/blob/master/projects/neurons/load_stringer_spontaneous.ipynb)  
[Stringer orientations](https://github.com/NeuromatchAcademy/course-content/blob/master/projects/neurons/load_stringer_orientations.ipynb)  
[Allen Institute 2p with behavior, SDK](https://github.com/NeuromatchAcademy/course-content/blob/master/projects/neurons/load_Allen_Visual_Behavior_from_SDK.ipynb)  
[Allen Institute 2p with behavior, simplified](https://github.com/NeuromatchAcademy/course-content/blob/master/projects/neurons/load_Allen_Visual_Behavior_from_pre_processed_file.ipynb)  
[Human Connectome Project FMRI](https://github.com/NeuromatchAcademy/course-content/tree/master/projects/fMRI)  
[Kay natural images FMRI](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/fMRI/load_kay_images.ipynb)  
[Bonner/Cichy etc FMRI](https://github.com/NeuromatchAcademy/course-content/tree/master/projects/fMRI)  
[ECoG](https://github.com/NeuromatchAcademy/course-content/tree/master/projects/ECoG)  
[Caltech social behavior](https://github.com/NeuromatchAcademy/course-content/blob/master/projects/behavior/Loading_CalMS21_data.ipynb)  
[IBL mouse decision making](https://github.com/NeuromatchAcademy/course-content/tree/master/projects/behavior)  

## [CRCNS](https://crcns.org/)
[Visual cortex](https://crcns.org/data-sets/vc)  
[Motor cortex](https://crcns.org/data-sets/motor-cortex)  
[Hippocampus](https://crcns.org/data-sets/hc)  

## [Janelia Figshare](https://janelia.figshare.com)
[Fly Connectome](https://www.janelia.org/project-team/flyem/hemibrain)  
[Hipposeq](https://hipposeq.janelia.org/)  
[MouseLight](https://www.janelia.org/project-team/mouselight)  
[OpenOrganelle](https://openorganelle.janelia.org/)  
[Stringer1: 10,000 V1 neurons with spontaneous activity](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622)  
[Stringer2: 10,000 V1 neurons with responses to 2,800 images](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_in_response_to_2_800_natural_images/6845348)  
[Stringer3: 20,000 V1 neurons with responses to fine orientations](https://janelia.figshare.com/articles/dataset/Recordings_of_20_000_neurons_from_V1_in_response_to_oriented_stimuli/8279387)  
[Steinmetz: 3,000 ephys neurons across brain with spontaneous activity](https://janelia.figshare.com/articles/dataset/Eight-probe_Neuropixels_recordings_during_spontaneous_behaviors/7739750)  

## Others
[Buzsaki lab webpage](https://buzsakilab.com/wp/database/)  
[EEG datasets](https://www.kaggle.com/search?q=EEG+in%3Adatasets)  
[BCI datasets](https://www.kaggle.com/search?q=BCI+in%3Adatasets)  
[Motor control](https://www.kaggle.com/fabriciotorquato/eeg-data-from-hands-movement)  

[Handwriting BCI dataset](https://www.kaggle.com/saurabhshahane/handwriting-bci). Relatively recent dataset (from Krishna Shenoy's lab) of motor cortex neural activity recordings during handwriting, for testing RNN models of text decoding.

Predict when or where an epileptic seizure will happen ([neurovista](https://www.epilepsyecosystem.org/neurovista-trial-1), [seizure_recognition](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)).

## Brain-Score

[About Brainscore](https://paperswithcode.com/dataset/brain-score) , [Main Website](http://www.brain-score.org/), [Tutorial](https://brain-score.readthedocs.io/en/latest/index.html), [preprint](https://www.biorxiv.org/content/10.1101/407007v2).

Overview: contains some datasets about behaviour and neural activity during image recognition tasks, as well as tools to compare similarity of ANN models to neural/behavioural data.
Potential research goals: comparing ANN activity and neural activity during image recognition. (*also need to look deeper at what datasets are available, might include more things). How does model architecture, learning rules influence similarity & performance?

Very preliminary [colab](https://colab.research.google.com/drive/1KUkwsbjDwLlmuoD3lPzmgTY1cYtSDpzR?usp=sharing).

## Allen Brain Observatory

The Allen Brain Observatory contains several relevant datasets.

[Older 2p data](http://observatory.brain-map.org/visualcoding/stimulus/natural_movies) with natural movie presentations

Very recent 2p data with behavior from NMA-CN: [Pre-processed subsample](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/neurons/load_Allen_Visual_Behavior_from_pre_processed_file.ipynb); [Entire dataset (SDK)](https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/neurons/load_Allen_Visual_Behavior_from_SDK.ipynb); [Getting started](https://allensdk.readthedocs.io/en/latest/visual_behavior_optical_physiology.html); A short description of the dataset can be watched [here](https://www.youtube.com/watch?v=3YP-GYvYnuA).

[Neuropixels data](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html) with [cheat sheet](https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/0f/5d/0f5d22c9-f8f6-428c-9f7a-2983631e72b4/neuropixels_cheat_sheet_nov_2019.pdf
). [Basic colab](https://colab.research.google.com/drive/1TPkgSzIPdyrAnQBAqZK9x7baYK6WvskK?usp=sharing) for fetching and parsing Allen Neuropixels dataset (very rudimentary right now).

## BCIAUT-P300
[Reference paper](https://www.frontiersin.org/articles/10.3389/fnins.2020.568104/full), [Dataset](https://www.kaggle.com/disbeat/bciaut-p300)

This a typical P300 dataset contains data from 15 subjects and 7 sessions for each subject.
