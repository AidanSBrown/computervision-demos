# What does a computer see?

Interactive demos for [the Catalyst](https://www.catalyst-magazine.com/) article explainer on computer vision in medicine. Demos follow the article structure, moving from how an image is stored on a computer to how computer vision models process images to detect features. Other articles I've written for the catalyst can be found [here](https://www.catalyst-magazine.com/articles?q=aidan+brown). 

[Find live demos here](https://aidansbrown.github.io/computervision-demos/)

| # | Demo | What you can do |
|---|------|-----------------|
| 1 | [Pixels are numbers](demos/01-pixels.html) | Zoom in until each pixel shows its RGB values. Change resolution and brightness levels, or load your own image. |
| 2 | [Filters find edges](demos/02-convolution.html) | Slide a 3 × 3 kernel across a lesion, step by step or as an animated scan. Edit the weights or types of filters. |
| 3 | [Layers build up shapes](demos/03-layers.html) | Watch a small hand-built CNN: edge filters → ReLU → pooling → combinations → summary numbers. Hover to see receptive fields. |
| 4 | [Drawing the line](demos/04-decision.html) | Plot demo 3's two output numbers for 160 lesions, then set the weights of a linear classifier with sliders or let gradient descent fit them. Sensitivity/specificity update live. |
| 5 | [Clever Hans shortcuts](demos/05-shortcuts.html) | Train a model in the browser where melanomas usually carry an ink mark, ruler or different camera. Then test it where they don't. |

## About the images

Every lesion is synthetic, drawn from a few parameters (size, border irregularity, darkness, color variation) loosely following the dermatology ABCD rule. That means:

- no patient data and no dataset licenses,
- every image can be reproduced from its seed,
- the artifacts in demo 5 can be switched on and off for the same lesion.

They are not realistic enough to train a real diagnostic model, as they are meant to be simpler explainers.

## Running on your own machine

To run locally, with no build step and no dependencies. Serve the folder with any localhost server:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>. Opening the HTML files directly also works in most browsers.

## How it's built

- `assets/common.js`: shared helpers: seeded random numbers, the synthetic lesion generator, color maps, slider wiring, and PNG export.
- `assets/network.js`: the small hand-built convolutional network. Demo 3 draws every stage of it; demo 4 plots its two output numbers.
- `assets/style.css`: the shared "demo window" layout.
- `demos/NN-*.html` + `demos/NN-*.js`: one page and one script per demo.

The model in demo 5 is logistic regression on raw 24 × 24 RGB pixels (1,728 inputs), trained with the Adam optimizer for 150 passes. It is the simplest possible neural network: one layer, no hidden units. Its weights can therefore be drawn directly as an image, which shows exactly what it learned to look for. Deep networks fall for the same shortcuts; they are just harder to inspect.

## References

- Sankarapandian et al. (2021). *A Pathology Deep Learning System Capable of Triage of Melanoma Specimens Utilizing Dermatopathologist Consensus as Ground Truth.* ICCV Workshops. [PDF](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/papers/Sankarapandian_A_Pathology_Deep_Learning_System_Capable_of_Triage_of_Melanoma_ICCVW_2021_paper.pdf)
- DeGrave, Janizek & Lee (2021). *AI for radiographic COVID-19 detection selects shortcuts over signal.* Nature Machine Intelligence 3, 610–619. [Link](https://www.nature.com/articles/s42256-021-00338-7)
- Badgeley et al. (2019). *Deep learning predicts hip fracture using confounding patient and healthcare variables.* npj Digital Medicine 2, 31. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550136/)
- Hagan, Demuth, Beale & De Jesús. *Neural Network Design* (2nd ed.) and its demos, the inspiration for the slider-driven layout.

## AI use statement

- Claude was used to generate the code for this project, as I do not know how to code in HTML or JavaScript. It additionally generated the first copy of this README and the captions in the demos. I have checked and edited all written sections in this project, and I don't use Claude to write thngs such as my catalyst articles. 
