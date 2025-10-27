# Face Detection Model

So recently I had been learning A.I / M.L and at this point have a pretty good understanding of how they work under the hood and the logic and maths behind them, so I wanted to make and train my 1st model with all the knowledge I have acquired so far and this is my attempt at making my 1st model from scratch.

## Why Face Detector?

Honestly, I was bored and I wanted something challenging and the classic number predicting model was too simple.

## Purpose of this readme.md?

This readme.md is a compilation of all my finding, research, intuition, and knowledge I acquired from this project. I am still new to A.I/M.L so below content might not be the best way or even a good one at that, also I might not know the right terms too. Sorry in advance 🙏.

## Intuition:

I knew I wanted to do something complex. So one random tuesday at 10pm I thought why not an object classifier and a face detector using a CNN to be specific, and now I just needed to figure out the input and features for it. Input was pretty simple just the RGB value at each pixel but features I had to figure out, so that day while trying to sleep I had an though why not 5 features x, y, height, width, confidence.

- x, y being the top left coordinate of where the rectangle should be placed.
- height, width being the size of the rectangle that encompasses the face.
- confidence number from (0, 1) representing models confidence in image containing a face for non face containing images.

## Findings

### Day 1:
- learning [tensorflow](https://www.tensorflow.org/).
- researching and finding dataset I could use.
- researching the feasibility of my idea. 

### Day 2:
- model implementation in tensorflow.
    - my original plan was to use dense layers but during my research I found out about convolution layers, which would work better for my use case. So I went a head learned more about them and ended up using them in most my hidden layer.
- picked and setup the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset dataset train the model with.
- tested impletion on very small training data like 10 images and everything worked as attend.

(after this point everything goes downhill 😭)

### Day 3:
- I wanted to train my model locally using my own GPU (RTX 3070).
    - this is is the point of no return.
    - out the box tensorflow should have GPU support, so what I thought
    - tensorflow cut GPU support for windows after [tensorflow 2.10](https://www.tensorflow.org/install/pip#windows-native)
    - my drivers were refusing to work
    - even GPT was making me go in circles
    - even in my dreams I was setting up my GPU
    - at this point my whole search history could be summarized with 2 words <b>cuda and tensorflow</b> but great learning experience tho.
- While my GPU and tensorflow were refusing to work together, I decided to refactor some code to make it easer to play around with model parameters.
- I find out I can use [WSL2](https://www.tensorflow.org/install/pip#windows-wsl2) to get tensorflow to register my GPU, no thanks to GPT, sometimes reading the documentation helps.

### Day 4
- train and test the model.
    - this is the 1st time I trained this model with about 10k images and It performed absolutely abysmally.
    - my intuition was to train it with even more images and it's not a good idea to load anymore images in my RAM so I decided to add an offset parameter to model_setup.py so I can load an existing model and train it on more new images.
    - this didn't fix the problem it still performed very poorly, so my intuition was that since I have confidence score as a feature but all my training data is of images with faces, so the model is just learning to maximize the confidence score since its always 1 for my training data, so I decided to add a new parameter to model_setup.py to add and control the quantity of images without faces and I did this by randomly inserting images with random RGB values for each pixel into the training data, this did improve the loss value but its was still extremely bad.

    