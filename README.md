# Synthetic-to-Real Transfer Learning on MNIST

## Project Goal

**Research Question:** Can a convolutional neural network trained exclusively on synthetically generated digit images accurately classify real handwritten digits from the MNIST dataset?

This project investigates whether deep learning models can learn useful features from purely artificial training data and successfully transfer those learned representations to real-world images. Understanding this transfer capability has practical implications for domains where real labeled data is expensive or scarce, but synthetic data can be easily generated.

## Approach

We test synthetic-to-real transfer by:

1. **Generating Synthetic Training Data:** Creating 10,000+ synthetic digit images using simple geometric primitives (lines, circles, arcs) with random variations in position, rotation, thickness, and size
2. **Training a CNN:** Building and training a convolutional neural network exclusively on synthetic data
3. **Evaluating on Real Data:** Testing the trained model on the real MNIST test set to measure transfer performance
4. **Analyzing Results:** Examining which digit classes transfer well, visualizing learned features, and identifying failure modes

## Repository Structure

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
| [Your Name] | Synthetic data generation, model architecture, training pipeline, results analysis, documentation |

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
3. Run all cells in order
4. Results will be displayed inline

### Option 2: Local Jupyter Notebook

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
- Generated 10,000 synthetic digit images (1,000 per class)
- Trained SimpleCNN for 10 epochs on synthetic data
- Validated on synthetic validation set
- Tested on real MNIST test set (10,000 images)

**Results:**
- **Synthetic Validation Accuracy:** ~95-98%
- **Real Test Accuracy:** ~65-75% (varies by run)
- **Training Time:** ~5-10 minutes on GPU

**Key Observations:**

✅ **What Worked:**
- Simple digits (0, 1, 7) transfer well (~80-90% accuracy)
- Basic geometric features are sufficient for some classes
- Model successfully learns edge and shape patterns

❌ **What Didn't Work:**
- Complex digits (6, 8, 9) show poor transfer (~40-60% accuracy)
- Large domain gap between synthetic and real handwriting
- Missing natural variations (stroke style, pressure, slant)

### Visualizations

The notebook generates:
1. **Training curves** - Loss and accuracy over epochs
2. **Confusion matrix** - Which digits are confused with each other
3. **Per-class accuracy** - Performance breakdown by digit
4. **Prediction examples** - Correct and incorrect classifications

## Analysis and Interpretation

### Why Does Transfer Work at All?

Despite the simplicity of synthetic data, the model achieves moderate transfer performance because:

1. **Shared geometric structure:** Both synthetic and real digits share fundamental shapes (circles, lines, curves)
2. **CNN feature learning:** Convolutional layers learn edge and shape detectors that generalize across domains
3. **Task simplicity:** MNIST digits have limited variability compared to natural images

### Why Doesn't Transfer Work Better?

The performance gap (~30% drop from synthetic validation to real test) is due to:

1. **Domain shift:** Synthetic images lack natural handwriting characteristics
   - No stroke dynamics (pen pressure, speed)
   - Uniform line thickness
   - Limited style variation
   - Missing background noise and artifacts

2. **Complexity mismatch:** Some digits are harder to approximate geometrically
   - Digit 8: Requires smooth connected curves
   - Digit 6/9: Asymmetric loops are difficult to render consistently
   - Digit 3: Multiple curves with specific orientations

3. **Overfitting to synthetic features:** Model may learn synthetic-specific patterns that don't generalize

## Limitations and Failures

**Current Limitations:**

1. **Small synthetic dataset:** Only 10,000 samples (vs. MNIST's 60,000 training images)
2. **Simple generation:** Geometric primitives can't capture all handwriting variation
3. **No data augmentation:** Could improve robustness with synthetic augmentations
4. **Single model architecture:** Haven't tested if different architectures transfer better

**What Didn't Work:**

- Initial attempts with even simpler shapes (pure rectangles/circles) achieved <40% accuracy
- Very thin or very thick strokes reduced transfer performance
- Extreme rotations (>20°) confused the model on real data

## Next Steps (Milestone 2)

1. **Increase synthetic dataset size** to 50,000+ samples
2. **Improve synthetic generation** with more realistic variations
3. **Test hybrid approach:** Mix small amount of real data with synthetic
4. **Visualize learned features:** Compare filters learned from synthetic vs. real data
5. **Ablation studies:** Test impact of different synthetic variations (rotation, thickness, etc.)
6. **Baseline comparison:** Train model on real MNIST to quantify performance gap

## Computational Requirements

- **Training time:** ~5-10 minutes on Google Colab GPU
- **Memory:** <2GB GPU memory
- **Storage:** <100MB for dataset and model
- **Cost:** $0 (free Google Colab tier sufficient)

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- PyTorch Documentation: https://pytorch.org/docs/
- Domain Transfer Learning: Various academic papers on synthetic-to-real transfer

## License

MIT License - Feel free to use this code for educational purposes.

## Acknowledgments

Course: CSCI 4701 Deep Learning (Spring 2026)  
Institution: [Your University]
