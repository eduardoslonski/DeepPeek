# DeepPeek

**DeepPeek** provides an interactive application to support interpretability research on Large Language Models.

This video gives an overall walkthrough of DeepPeek, showing all its functionalities:

https://github.com/eduardoslonski/DeepPeek_private/assets/121900778/ef148fdf-1d41-4559-be71-237fc82fac83

### Tools

- **Sample Selector**: Choose samples from a range of datasets or write new samples.
- **Predictions**: Top 5 predictions with the final softmaxed logits values displayed with bars (correct prediction in blue).
- **Attention and Opposite Attention**: Shows attention from selected token to previous ones, and future tokens to selected token, respectively.
- **Cross Entropy Loss**
- **Similarities**:
  - **Tokens**: Cosine similarity of the activation vector of the selected token against all other token vectors in the selected layer (and attention head, if applicable).
  - **Previous**: Cosine similarity of the activation vector of the selected token against the same vector in the previous layer.
  - **Previous Residual**: Cosine similarity of the activation vector of the selected token against the activation vector from the immediate previous residual stream (e.g., when there is a skip connection in the block at attention @ dense + x).
- **Activations**:
  - **Sum**: Sum of the absolute values of the selected activation vector. Particularly useful in `o_mm_dense` and `dense_attention` due to dynamics with small vectors of "nothing" tokens.
  - **Value**: Specific activation value, e.g., the 10th activation of `mlp_4_to_h`.
- **Layer and Attention Head Selectors**: Select by typing directly or using arrows.
- **Scatter Plots**: Visualize all types of activations. Select and highlight individual activation dimensions for detailed analysis. Selected tokens are highlighted on the plot.
- **Histograms**: Display histograms for all activation types. The median histogram, shown in orange, serves as the base for all tokens and facilitates accurate comparisons. The histogram of the selected token is displayed in blue. Outliers are identified using the IQR method with a factor of 3, performed individually for negative and positive values due to skewed distributions and model dynamics. Specific activations can be highlighted and will be shown as a blue vertical line.

### Features

- **Opacity Selector**
- **Tooltips**: Display values and indices for each token based on the selected tool.
- **Show/Hide `\n` Token**: Assists when copying the sample.
- **[Alt] Multiply Selector for Attention**
- **RoPE**: Full, Applied, or Not Applied to isolate the influence of RoPE on Q and K.
- **Normalization**: Default (None), Normalized, or Normalized excluding outliers.
- **Activation Limits Highlight**: Display minimum, divider, and maximum values used in calculating background colors for activation analysis.
- **Commands**:
  - Download the activation vector of a selected token using `Command` + click.
  - Highlight the selected token by holding `CTRL`.
  - Make attention more visible by holding `ALT`.

### Samples

- **Sample Writing and Selection**: Write or choose samples from the list.
- **Multiple Sample Analysis**: Work with multiple samples simultaneously for comparative analysis.
- **Pre-included Samples**: Repository contains 10,000 samples from various datasets such as AI4Code, arXiv, Enwiki, GitHub Issues, Gutenberg, Opensubtitles, OtherWiki, PubMed, Slimpajama, and TheStack. Each dataset includes 1,000 samples.
- **Slimpajama Samples**: Included in a stacked and randomized format, mimicking real-world training setups. This arrangement ensures that the samples reflect the variability and complexity expected in actual model training scenarios.
- **Sample Selector**: 10 pre-tokenized samples from each dataset are available in DeepPeek, with relevant titles. More samples can be processed as needed.

### Model

- Analyze using **Pythia** models of various sizes (70M, 160M, 410M, 1.0B, 1.4B, 2.8B, 6.9B, 12B) available [here](https://huggingface.co/EleutherAI). DeepPeek adapts dynamically to model attributes such as dimensionality, number of layers, and attention head dimensions.

### Getting Started

DeepPeek uses Python + Flask for the backend and TypeScript + React for the frontend.

- **Install Frontend Dependencies**: `npm install`.
- **Install Backend Dependencies**: `pip install -r requirements.txt`.

Run the backend with `python run.py` and the frontend with any `npm run` command.
