# MSED: A multi-modal sleep event detection model for clinical sleep analysis

This repo contains inference code for running event predictions using PSG signals as described in *A. N. Zahid, P. Jennum, E. Mignot and H. B. D. Sorensen, "MSED: A Multi-Modal Sleep Event Detection Model for Clinical Sleep Analysis," in IEEE Transactions on Biomedical Engineering, vol. 70, no. 9, pp. 2508-2518, Sept. 2023, doi: 10.1109/TBME.2023.3252368*

## Installation
The project can be installed by running the following commands:
```bash
git clone https://github.com/neergaard/msed.git
cd msed
make project
```
which will create a new Python 3.11 environment with all the required packages.

If you want to use this package in your own environment, you can use the following:
```bash
git clone https://github.com/neergaard/msed.git
cd msed
pip install -e .
```
Note that the last command requires the proper packages be installed in your environment beforehand. See the [requirements](requirements.txt) for details.

## Running inference
After installing the package, you can get event predictions using the `msed-predict` terminal command:
```bash
msed-predict --data-path <path to directory containing EDF files> \
             --target-dir <directory for outputs, predictions etc.> \
             --match-pattern <string pattern to match filenames (optional)> \
             --device <device to use (cpu, gpu)> \
             --model-path <path to trained model directory. Default `model/splitstream`>
```
A .csv file will be created in the `target-dir` for each subject file found in `data-path` along with a `status.csv` containing an overview of which files had predicted events and which files have failed processing. The prediction CSV files will contain start and stop indices for each predicted events.

### Channel mapping
The model will search `data-dir` for relevant files and will thereafter create a channel map object containing the mapping between channel labels found in the EDF files and the required channels in the model. The program will ask the user to assign indices for each channel category, and multiple channel labels can be assigned to the same category, as shown in the example below using 23 EDFs from MASS S2:
```bash
msed-predict --data-path data/raw/mass/SS1 --match-pattern PSG --target-dir outputs/mass/SS1 --device cpu
[14:58:01] INFO     Usage: /Users/aneol/miniconda3/envs/msed/bin/msed-predict --data-path data/raw/mass/SS1
                    --match-pattern PSG --target-dir outputs/mass/SS1 --device cpu                            argument_parser.py:40

           INFO     Settings:                                                                                 argument_parser.py:41
           INFO     ---------                                                                                 argument_parser.py:42
           INFO     data_path       data/raw/mass/SS1                                                         argument_parser.py:47
           INFO     device          cpu                                                                       argument_parser.py:47
           INFO     match_pattern   PSG                                                                       argument_parser.py:47
           INFO     model_path      models/splitstream                                                        argument_parser.py:47
           INFO     target_dir      outputs/mass/SS1                                                          argument_parser.py:45

           INFO     Saving predictions to "outputs/mass/SS1"                                                  argument_parser.py:35
           INFO     Loading model configuration from "models/splitstream"...                                   predict_events.py:94
           INFO     Getting relevant EDF files from "data/raw/mass/SS1"...                                     predict_events.py:29
           INFO     Found 23 EDF files.                                                                        predict_events.py:50
           INFO     Creating channel map (if it does not exist)...                                            predict_events.py:101
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:00<00:00, 23.41it/s]

Enter acceptable channel indices to use for the given identifier.
Use spaces to separate multiple indices.
Total number of EDFs in directory: 23

0.    23 ECG ECGI             1.    18 EEG A2-CLE           2.    18 EEG C3-CLE           3.     5 EEG C3-LER
4.    18 EEG C4-CLE           5.     5 EEG C4-LER           6.    18 EEG Cz-CLE           7.     5 EEG Cz-LER
8.    18 EEG F3-CLE           9.     5 EEG F3-LER           10.   18 EEG F4-CLE           11.    5 EEG F4-LER
12.   18 EEG F7-CLE           13.    5 EEG F7-LER           14.   18 EEG F8-CLE           15.    5 EEG F8-LER
16.    3 EEG Fp1-LER          17.    3 EEG Fp2-LER          18.   18 EEG Fz-CLE           19.    5 EEG Fz-LER
20.   18 EEG O1-CLE           21.    5 EEG O1-LER           22.   18 EEG O2-CLE           23.    5 EEG O2-LER
24.   18 EEG P3-CLE           25.    5 EEG P3-LER           26.   18 EEG P4-CLE           27.    5 EEG P4-LER
28.   18 EEG Pz-CLE           29.    5 EEG Pz-LER           30.   18 EEG T3-CLE           31.    5 EEG T3-LER
32.   18 EEG T4-CLE           33.    5 EEG T4-LER           34.   18 EEG T5-CLE           35.    5 EEG T5-LER
36.   18 EEG T6-CLE           37.    5 EEG T6-LER           38.   23 EMG Ant Tibial L     39.   23 EMG Ant Tibial R
40.   23 EMG Chin1            41.   23 EMG Chin2            42.   23 EMG Chin3            43.   23 EOG Left Horiz
44.   23 EOG Right Horiz      45.   23 Resp Belt Abdo       46.   20 Resp Belt Thor       47.   23 Resp Cannula
48.   23 Resp Thermistor      49.   23 SaO2 SaO2

A1:
Selected: []
A2: 1
Selected: ['EEG A2-CLE']
C3: 3 2
Selected: ['EEG C3-LER', 'EEG C3-CLE']
C4: 5 4
Selected: ['EEG C4-LER', 'EEG C4-CLE']
EOGL: 43
Selected: ['EOG Left Horiz']
EOGR: 44
Selected: ['EOG Right Horiz']
EOGRef:
Selected: []
Chin: 41
Selected: ['EMG Chin2']
ChinRef: 40 42
Selected: ['EMG Chin1', 'EMG Chin3']
LegL: 38
Selected: ['EMG Ant Tibial L']
LegR: 39
Selected: ['EMG Ant Tibial R']
NasalP: 47
Selected: ['Resp Cannula']
Thor: 46
Selected: ['Resp Belt Thor']
Abdo: 45
Selected: ['Resp Belt Abdo']
```


## Citation
If you use this work, please cite the following publication:
- A. N. Zahid, P. Jennum, E. Mignot and H. B. D. Sorensen, "MSED: A Multi-Modal Sleep Event Detection Model for Clinical Sleep Analysis," in IEEE Transactions on Biomedical Engineering, vol. 70, no. 9, pp. 2508-2518, Sept. 2023, doi: 10.1109/TBME.2023.3252368

You can also use the following BibTex citation:
```bibtex
@article{Zahid2023,
  title = {{{MSED}}: A Multi-Modal Sleep Event Detection Model for Clinical Sleep Analysis},
  shorttitle = {{{MSED}}},
  journaltitle = {IEEE Trans. Biomed. Eng.}
  author = {Zahid, Alexander Neergaard and Jennum, Poul and Mignot, Emmanuel and Sorensen, Helge B. D.},
  volume = {70},
  number = {9}
  year = {2023},
  journal = {IEEE Transactions on Biomedical Engineering},
  pages = {2508-2518},
  doi = {10.1109/TBME.2023.3252368},
}
```
