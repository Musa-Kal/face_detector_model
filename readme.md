# Face Detector Model

So recently I had been learning A.I / M.L and at this point have a pretty good understanding of how they work under the hood and the logic and maths behind them, so I wanted to make and train my 1st model with all the knowledge I have acquired so far and this is my attempt at making my 1st model from scratch.

## Why Face Detector?

Honestly, I was bored and I wanted something challenging and the classic number predicting model was too simple.

## Purpose of this readme.md?

This readme.md is a compilation of all my finding, research, intuition, and knowledge I acquired from this project. I am still new to A.I/M.L so below content might not be the best way or even a good one at that, also I might not know the right terms too. Sorry in advance 🙏.

## Intuition:

I knew I wanted to do something complex. So one random tuesday at 10pm I thought why not an object classifier and a face detector to be specific, and now I just needed to figure out the input and features for it. Input was pretty simple just the RGB value at each pixel but features I had to figure out, so that day while trying to sleep I had an though why not 5 features x, y, height, width, confidence.

- x, y being the top left coordinate of where the rectangle should be placed.
- height, width being the size of the rectangle that encompasses the face.
- confidence number from (0, 1) representing models confidence in image containing a face for non face containing images.

## Findings

Day 1:
- learning tensorflow.
- researching and finding dataset I could use.
- researching the feasibility of my idea. 