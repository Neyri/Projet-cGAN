# Projet c-GAN / UNIT 
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

Welcome to the C-GAN / UNIT Project. 

Multiple  computer  vision  problems  can  be  posed  astransforming one input image into an output one (edge de-tection,  image  enhancement,  colorization,  restoration  ...).As such, the Image to Image translation refers to the map-ping from one domain to another and it’s an active area ofresearch given the many possibilities that it has. For exam-ple,  let’s consider super-resolution,  which  consists  of up-sampling a low resolution image to a realistic higher resolu-tion one. We already know that going to a lesser resolutionis a trivial problem but upsampling requires inferring un-known details.  Finding a framework that can perform sucha task would allow you to watch your favorite movie fromthe 70s in a higher quality or retrieve a picture from thatvery old phone you had 20 years ago and make it nice. The problem can be approached both as a supervised and unsupervised  problem.  

We chose to reproduce the results from two papers who approached the image-to-image translation problem from both settings : 
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) and [UNIT: UNsupervised Image-to-image Translation Networks](https://arxiv.org/pdf/1703.00848.pdf)

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
