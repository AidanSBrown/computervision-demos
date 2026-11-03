# What does a computer see?

Interactive demos for an explainer on computer vision in medicine. They walk from "an image is a grid of numbers" to "a neural network makes a diagnosis", then show how a model can learn the wrong clue (the Clever Hans effect).

**Live demos:** `https://<your-username>.github.io/<repo-name>/` (after enabling GitHub Pages, see below)

| # | Demo | What you can do |
|---|------|-----------------|
| 1 | [Pixels are numbers](demos/01-pixels.html) | Zoom in until each pixel shows its RGB values. Change resolution and brightness levels, or load your own image. |
| 2 | [Filters find edges](demos/02-convolution.html) | Slide a 3 × 3 kernel across a lesion, step by step or as an animated scan. Edit the weights. |
| 3 | [Layers build up shapes](demos/03-layers.html) | Watch a small hand-built CNN: edge filters → ReLU → pooling → combinations → summary numbers. Hover to see receptive fields. |
| 4 | [Drawing the line](demos/04-decision.html) | Plot demo 3's two output numbers for 160 lesions, then set the weights of a linear classifier with sliders or let gradient descent fit them. Sensitivity/specificity update live. |
| 5 | [Clever Hans shortcuts](demos/05-shortcuts.html) | Train a model in the browser where melanomas usually carry an ink mark, ruler or different camera. Then test it where they don't. |

Article-ready screenshots are in [`figures/`](figures/).

## About the images

Every lesion is **synthetic**, drawn from a few parameters (size, border irregularity, darkness, color variation) loosely following the dermatology ABCD rule. That means:

- no patient data and no dataset licenses,
- every image can be reproduced from its seed,
- the artifacts in demo 5 can be switched on and off for the exact same lesion.

They are not realistic enough to train a real diagnostic model, and they aren't meant to be.

## Running locally

No build step and no dependencies. Serve the folder with any static server:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>. Opening the HTML files directly also works in most browsers.

## Making figures

- **Save figure** (top-right of each demo) downloads a clean PNG of the main panels.
- Adding `?clean` to any demo URL hides the site navigation, for tidy screenshots.
- Demo 4 accepts `?trained` to show the fitted line, its training trail and the error curve.
- Demo 5 accepts settings in the URL, e.g. `05-shortcuts.html?kind=ruler&strength=90` (`kind` = `ink`, `ruler` or `tint`; `strength` = 0–100).

The PNGs in `figures/` were captured with headless Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
  --window-size=1280,1000 --force-device-scale-factor=2 --virtual-time-budget=15000 \
  --screenshot=figures/demo5-ink.png "http://localhost:8000/demos/05-shortcuts.html?clean&kind=ink&strength=90"
```

## Publishing on GitHub Pages

1. Create a repository and push this folder to it.
2. In the repository, go to **Settings → Pages**, set *Source* to "Deploy from a branch", branch `main`, folder `/ (root)`.
3. The site appears at `https://<your-username>.github.io/<repo-name>/` within a minute or two.

## How it's built

- `assets/common.js`: shared helpers: seeded random numbers, the synthetic lesion generator, color maps, slider wiring and PNG export.
- `assets/network.js`: the small hand-built convolutional network. Demo 3 draws every stage of it; demo 4 plots its two output numbers.
- `assets/style.css`: the shared "demo window" layout.
- `demos/NN-*.html` + `demos/NN-*.js`: one page and one script per demo.

The model in demo 5 is logistic regression on raw 24 × 24 RGB pixels (1,728 inputs), trained with Adam for 150 passes. It is the simplest possible neural network: one layer, no hidden units. Its weights can therefore be drawn directly as an image, which shows exactly what it learned to look for. Deep networks fall for the same shortcuts; they are just harder to inspect.

## References

- Sankarapandian et al. (2021). *A Pathology Deep Learning System Capable of Triage of Melanoma Specimens Utilizing Dermatopathologist Consensus as Ground Truth.* ICCV Workshops. [PDF](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/papers/Sankarapandian_A_Pathology_Deep_Learning_System_Capable_of_Triage_of_Melanoma_ICCVW_2021_paper.pdf)
- DeGrave, Janizek & Lee (2021). *AI for radiographic COVID-19 detection selects shortcuts over signal.* Nature Machine Intelligence 3, 610–619. [Link](https://www.nature.com/articles/s42256-021-00338-7)
- Badgeley et al. (2019). *Deep learning predicts hip fracture using confounding patient and healthcare variables.* npj Digital Medicine 2, 31. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550136/)
- Hagan, Demuth, Beale & De Jesús. *Neural Network Design* (2nd ed.) and its demos, the inspiration for the slider-driven layout.
