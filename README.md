# Synthetic-to-Real Transfer Learning on MNIST

## Project Goal

**Research Question:** Can a convolutional neural network trained exclusively on synthetically generated digit images accurately classify real handwritten digits from the MNIST dataset?

This project investigates whether deep learning models, especially CNNs, can learn useful features from entirely artifical training data and successfully generalize these learned representations to MNIST data images. Understanding this transfer capability of deep neural networks has real-life practical implications for fields where real data may be scarce or expensive, and generating synthetic data might be easier. In this case, whether such artifical datasets (that look effortless and slam-dunk) can transfer their features to real-life examples.

## Approach

I test synthetic-to-real transfer by:

1. Generating Synthetic Training Data: Creating 10,000 synthetic digit images using simple geometric primitives (lines, circles, arcs) with random variabilities in their features (i.e line thickness)
2. Training a CNN: Building and training CNN exclusively on this synthetic data
3. Evaluating on Real Data: Testing the trained model on the real MNIST test set to measure transfer performance
4. Analyzing Results: Examining which digits transfer better or worse, visualizing learned features, and identifying failures (failure modes)

## Repository Structure 

### (This was done by AI, I gave the structure and it gave me this nice looking tree)

```
synthetic-to-real-mnist/
├── README.md                    # This file
├── milestone1_notebook.ipynb    # Main graded notebook
├── synthetic_data.py            # Synthetic digit generation
├── model.py                     # CNN architecture definition  
├── train.py                     # Training utilities
├── requirements.txt             # Python dependencies
└── results/                     # Generated plots and outputs
```

## Team Members and Responsibilities

| Name | Responsibilities |
|------|-----------------|
| [Isgandar Jafarli] | Synthetic data generation, model architecture, training pipeline, results analysis, documentation |

I basically did everything since it is an individual project for me

## Setup and Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- Pillow
- scikit-learn
- tqdm

### Install Dependencies

```bash
pip install -r requirements.txt
```



Or for Google Colab (all dependencies pre-installed except tqdm):
```bash
pip install tqdm
```

## How to Run

### Option 1: Google Colab (Recommended)

1. Open Google Colab: https://colab.research.google.com/
2. Upload `milestone1_notebook.ipynb` or clone this repository
3. Change the username to your github username to sychronize with other necessary files (otherwise, it won't run properly)
4. Run all cells in order
5. Results will be displayed inline

### Option 2: Local Jupyter Notebook 

#### This pieace fo text about how to run on Jupyter was AI-aided in writing since this is not my specialization, 
#### I wanted it to look accurate :D 

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/synthetic-to-real-mnist.git
cd synthetic-to-real-mnist

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook milestone1_notebook.ipynb
```

### Option 3: Command Line

```bash
# Generate synthetic data
python synthetic_data.py

# Train and evaluate (coming in Milestone 2)
python main.py
```

## Milestone 1 Results

### Experiment 1: Synthetic Training and Real Testing

**Setup:**
- Generated 10,000 synthetic digit images (1,000 per class(0-9))
- Trained Simple CNN for 10 epochs on synthetic data
- validated on synthetic validation set
- Tested on real MNIST test set (10,000 images, again)

**Results:**
- **Synthetic Validation Accuracy:** up to 95%
- **Real Test Accuracy:** ~38-49% (varies by run) (most important one)
- **Training Time:** ~5-10 minutes on GPU

**Key Observations:**

✅ **What Worked:**
- Simple digits (0, 1, 7) transfer well (~80-90% accuracy)
- Basic geometric features are sufficient for some classes
- Model successfully learns edge and shape patterns (i took advantage of sliding windows, specialized kernels and slight padding)

❌ **What Didn't Work:**
- Complex digits (3, 8, 9) show poor transfer (~5-15% accuracy)
- Large domain gap between synthetic and real handwriting 
- Missing natural variations (stroke style, pressure, slant)
- Especially that the circles in 8 weren't connected (will be fixed by the next project)

### Visualizations

The notebook generates:
1. Training curves - Loss and accuracy over epochs
2. Confusion matrix - Which digits are confused with each other (please look at it yourself, very interesting results)
3. Per-class accuracy - Performance breakdown by digit
4. Prediction examples - Correct and incorrect classifications

## Analysis and Interpretation

### Why Does Transfer Work at All?

Despite the simplicity of synthetic data, the model achieves moderate prediction accuracy due to:

1. **Shared geometric structure:** Both synthetic and real digits share fundamental shapes (circles, lines, curves)
2. **CNN feature learning:** Convolutional layers learn edge and shape detectors that generalize well
3. **Task simplicity:** MNIST digits have limited variability compared to natural images, which makes it easier to learn patterns

### Why Doesn't Transfer Work Better?

The performance gap (~60% drop from synthetic validation to real test) is due to:

1. **Domain shift:** Synthetic images lack natural handwriting characteristics
   - No stroke dynamics (pen pressure, speed)
   - Uniform line thickness
   - Limited variations in style
   - Missing background noise and artifacts

2. **Complexity mismatch:** Some digits are harder to approximate geometrically
   - Digit 8: Requires smooth connected curves (slanted infinity symbol)
   - Digit 6/9: Asymmetric loops are difficult to render consistently (I tried, trust me)
   - Digit 3: Multiple curves with specific orientations

3. **Overfitting to synthetic features:** Model may learn synthetic-specific patterns that don't generalize (I will try to improve this by Milestone 2)

## Limitations and Failures

**Current Limitations:**

1. **Small synthetic dataset:** Only 10,000 samples
2. **Simple generation:** Geometric primitives can't capture all handwriting variation
3. **No data augmentation:** Could improve robustness with synthetic augmentations
4. **Single model architecture:** Haven't tested if different architectures transfer better (will be done by Milestone 2)

**What Didn't Work:**

- Initial attempts with even simpler shapes (pure rectangles/circles) achieved <15% accuracy
- Very thin or very thick strokes reduced transfer performance
- Extreme rotations (>20°) confused the model on real data (don't ask why I tried that lol)

## Next Steps (Milestone 2)

1. **Increase synthetic dataset size** to 50,000+ samples
2. **Improve synthetic generation** with more realistic variations
3. **Test hybrid approach:** Mix small amount of real data with synthetic
4. **Visualize learned features:** Compare filters learned from synthetic vs. real data
5. **Ablation studies:** Test impact of different synthetic variations (rotation, thickness, etc.)
6. **Baseline comparison:** Train model on real MNIST to quantify performance gap
7. **Synthetic Digit Improvement** Improve the way, especially, how some digits look

## Computational Requirements

- **Training time:** ~5-10 minutes on Google Colab GPU
- **Memory:** <2GB GPU memory
- **Storage:** <100MB for dataset and model
- **Cost:** $0 (free Google Colab tier sufficient)

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- PyTorch Documentation: https://pytorch.org/docs/
- Domain Transfer Learning: Various academic papers on synthetic-to-real transfer (all read by AI and narrated to me)

## License

Feel free to use this code for educational purposes.

## Acknowledgments

Course: CSCI 4701 Deep Learning (Spring 2026)  
Institution: ADA University
Student: Isgandar Jafarli
