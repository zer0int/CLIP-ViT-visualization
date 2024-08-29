![generate-some-vit-viz](https://github.com/zer0int/CLIP-ViT-visualization/assets/132047210/6399dc8d-9990-460c-866c-81ef0a25db05)

# CLIP-ViT-Visualization
Standing on the shoulders of giants:
- Based on [github.com/hamidkazemi22/vit-visualization](https://github.com/hamidkazemi22/vit-visualization)
- This repo is a lightweight, CLIP ViT feature visualization only, implementation thereof.
- Contains ALL (!) CLIP models + some speed optimization. Requires CUDA.
- Uses [OpenAI / CLIP](https://github.com/openai/CLIP)

## Info

- Check / install `requirements.txt`
- Check the comments in "run_visualization.py" -> Easy lightweight configuration
- Configure: model, optimizer, single-feature vs. multi-feature/layer, intermediate steps, ...
- From the console, use "python run_visualization.py"
- Saves output in neatly organized and named way (subfolders)
- 29-AUG-204: Added script for deterministic results.


## Warning about CLIP feature visualizations

- CLIP has been trained on "pretty much the entire internet". 
- One neuron gets you a cat, a rose; another may encode explicit / sensitive / offensive / violent content. 
- Use responsibly / at your own discretion.
- For more information, refer to the [CLIP Model Card](https://github.com/openai/CLIP/blob/main/model-card.md).
