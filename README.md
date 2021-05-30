### Tracking Soccer Players in low quality video

Elói Martins

José Henrique Brito

![tracking results](https://github.com/Eloi-14855/tracking_soccer_players/blob/tracking_soccer/Example_of_results.jpg?raw=true)

## Description
 
This repository contains:
- dataset with video in 3 quality levels with player tracking groundtruth data
- a prototype application for tracking soccer players in low quality video

If you use this work please cite:

Eloi Martins, José Henrique Brito (2021), Soccer Player Tracking in Low Quality Video, arXiv 2021 https://arxiv.org/abs/2105.10700

The code relies on the use of Detectron2 and Tracktor

# Instalation of Detectron 2
The file "Detectron " is the script that is related to the creation of the detection dataset. is better to be implemented on a new project because of the version differences to Tracktor.

The instalation of Detectron2 on windows can be done like explained here : https://dgmaxime.medium.com/how-to-easily-install-detectron2-on-windows-10-39186139101c

# Instalation of Tracktor
Tracktor instalation:

- Tracktor installation needs to follow a specific way to work on Pycharm/Windows

- Create a virtual invironment on anaconda:

- Install matplotlib with : ' python -m pip install -U matplotlib==3.1.1 '

- Install torch with : ' pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html '

- (clear the requirements on TXT file  that are already installed on environment)

- Install requirements with: ' pip install -r requirements.txt '

- With all packages isntalled create a virtual environment on pycharm

- In "settings-project interpeter- add" add the conda environment where the packages are installed

- Confirm if the packages are all installed

- Through the terminal install my git hub: " git clone https://github.com/Eloi-14855/tracking_soccer_players.git "

- Next after make " cd " to the directory where installed do : " pip install -e . "
- The folder "data" and "Output" need to be download in order to have our models pretrained in our dataset. it can be downloaded the original dataset and output from https://github.com/phil-bergmann/tracking_wo_bnw, or our:

###### Data
https://drive.google.com/file/d/1j4Agbv4ckyN3YQx6jsxfu6VC4DQAOUuL/view?usp=sharing

###### Output
https://drive.google.com/file/d/1DhASFRgtnxoawvC8F05na4-PFLTEMaEx/view?usp=sharing


- After all run " test_trackor" , maube it will ask you some modules like Tqmd , sacred ,opencv-python (cv2). if that problem occur go to requirements.txt, check the version needed for each and install 1 by one. 

- The paths need to be updated to your machine in "tracktor.ymal" and " test_tracktor" files.





