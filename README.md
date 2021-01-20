![alt text](https://github.com/Eloi-14855/tracking_soccer_players/blob/tracking_soccer/Example_of_results.jpg?raw=true)

![alt text](https://ibb.co/FK9zsbs)

The file "Detectron " is the script that is related to the creation of the detection dataset. is better to be implemented on a new project because of the version differences to Tracktor.
The instalation of Detectron2 on windows can be done like explained here : https://dgmaxime.medium.com/how-to-easily-install-detectron2-on-windows-10-39186139101c


Tracktor isntalation:
Tracktor isnstallation needs to follow a specific way to work on pycharm windowns
create a virtual invironment on anaconda:
install matplotlib with : ' python -m pip install -U matplotlib==3.1.1 '
install torch with : ' pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html '
(clear the requirements on TXT file  that are already installed on environment)
install requirements with: ' pip install -r requirements.txt '
with all packages isntalled create a virtual environment on pycharm
in "settings-project interpeter- add" add the conda environment where the packages are installed
confirm if the packages are all installed
through the terminal install my git hub: " https://github.com/Eloi-14855/tracking_soccer_players.git "
next after make " cd " to the directory where installed do : " pip install -e . "


