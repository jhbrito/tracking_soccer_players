### In this repositorie we present the code part of our project " Tracking soccer players in low-quality videos"

![alt text](https://github.com/Eloi-14855/tracking_soccer_players/blob/tracking_soccer/Example_of_results.jpg?raw=true)


# Instalation of Detectron 2
The file "Detectron " is the script that is related to the creation of the detection dataset. is better to be implemented on a new project because of the version differences to Tracktor.

The instalation of Detectron2 on windows can be done like explained here : https://dgmaxime.medium.com/how-to-easily-install-detectron2-on-windows-10-39186139101c

# instalation of Tracktor
Tracktor instalation:

- Tracktor isnstallation needs to follow a specific way to work on pycharm windowns

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

##### Data



##### Output


- After all run " test_trackor" , maube it will ask you some modules like Tqmd , sacred ,opencv-python (cv2). if that problem occur go to requirements.txt, check the version needed for each and install 1 by one. 

- The paths need to be updated to your machine in "tracktor.ymal" and " test_tracktor" files.





