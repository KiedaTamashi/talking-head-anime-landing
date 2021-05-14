# Improved version for "Talking Head Anime from a Single Image"  
  
This repository contains code for two applications that make use of the neural network system in the [Talking Head Anime from a Single Image](http://pkhungurn.github.io/talking-head-anime/) project:  
  
* The *manual poser* allows the user to pose an anime character by manually manipulating sliders.
* The *puppeteer* makes an anime character imitate the head movement of the human capture by a webcam feed.

## Hardware Requirements

As with many modern machine learning projects written with PyTorch, this piece of code requires **a recent and powerful Nvidia GPU** to run. I have personally run the code on a Geforce GTX 1080 Ti and a Titan RTX.

Also, the peppeteer tool requires a webcam.

## Installation and Usage.
Download the release wheel "talkingHeadAnimeLanding-0.1-py3-none-any.whl"
Install it with pip in your environment
> pip install talkingHeadAnimeLanding-0.1-py3-none-any.whl

Run the application by simple two line, and you will see a user-friendly UI.
> from talkingHeadAnime import puppeteer 
>
>  puppeteer.run()

## Dependencies
* Python >= 3.6
* pytorch >= 1.4.0
* dlib >= 19.19
* opencv-python >= 4.1.0.30
* pillow >= 7.0.0
* numpy >= 1.17.l2
* waifulabs

------------------------------------------Following is the original guide-----------------------------------------------

## Recreating Python Environment with Anaconda

If you use [Anaconda](https://www.anaconda.com/), you also have the option of recreating the Python environment that can be used to run the demo. Open a shell and change directory to the project's root. Then, run the following command:

> `conda env create -f environment.yml`

This should download and install all the dependencies. Keep in mind, though, that this will require several gigabytes of your storage. After the installation is done, you can activate the new environment with the following command:

> `conda activate talking-head-anime`

Once you are done with the environment, you can deactivate it with:

> `conda deactivate`

## Prepare the Data

After you cloned this repository to your machine's storage, you need to download the models: 

* Download the main models from [this link](https://drive.google.com/open?id=1ajHViqyLDKFKfBtGPE5cbSGcMNa8rz8k). Unzip the file into the `data` directory under the project's root. The models are released separately with the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/legalcode).
* Download `shape_predictor_68_face_landmarks.dat` and save it to the `data` directory. You can download the bzip archive from [here](https://github.com/davisking/dlib-models). Do not forget to uncompress.

Once the downloading is done, the data directory should look like the following:

```
+ data
  + illust
    - placeholder.txt
    - waifu_00_256.png
    - waifu_01_256.png
    - waifu_02_256.png
    - waifu_03_256.png
    - waifu_04_256.png
  - combiner.pt
  - face_morpher.pt
  - placeholder.txt
  - shape_predictor_68_face_landmarks.dat
  - two_algo_face_rotator.pt
```

## Running the Program

Change directory to the root directory of the project. To run the manual poser, issue the following command in your shell:

> `python app/manual_poser.py`

To run the puppeteer, issue the following command in your shell:

> `python app/puppeteer.py`

## Disclaimer

While the author is an employee of Google Japan, this software is not Google's product and is not supported by Google.

The copyright of this software belongs to me as I have requested it using the <a href="https://opensource.google/docs/iarc/">IARC process</a>. However, one of the condition for the release of this source code is that the publication of the "Talking Head Anime from a Single Image" be approved by the internal publication approval process. I requested approval on 2019/11/17. It has been reviewed by a researcher, but has not been formally approved by a manager in my product area (Google Maps). I have decided to release this code, bearing all the risks that it may incur.

I made use of [a face tracker code implemented by KwanHua Lee](https://github.com/lincolnhard/head-pose-estimation) to implement the puppeteer tool.
