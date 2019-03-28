# Projet c-GAN
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

Welcome to the C-GAN Project

## Getting started
Make sure you have PyTorch installed on your virtual environment.
This can be done by pasting the following command on your terminal: 
- `python -c "import torch; print(torch.__version__)"`

##### Configure your `env.py`
Create a file named `env.py` in the project root dir. Inside it set the following environment variables: 

- `DISCRIMINATOR_PATH` — the path from root directory to the discriminator .pth file
- `GENERATOR_PATH` — the path from root directory to the generator .pth file

Create a directory containing the inputs in the project root dir.

##### File Permissions

Make the script executable the first time you run it:
- `chmod +x demo.py`


##### Commands
- `-h` — show this help message and exit
- `-in` — path to input file

##### Options
- `-n` — number of data to display, 1 by default
- `--save` — store the data on a specific folder, False by default
