# PyEhsa Examples

This directory contains examples demonstrating how to use PyEhsa for Emerging Hot Spot Analysis.

## Available Examples

### 1. Basic Usage Script (`basic_usage.py`)

A comprehensive Python script that demonstrates:
- Creating synthetic spatio-temporal data
- Running EHSA analysis with PyEhsa  
- Interpreting results and classifications
- Using preprocessing functions

**Run the example:**
```bash
cd examples/
python basic_usage.py
```

**What you'll see:**
- Synthetic dataset creation (16 regions, 8 time periods)
- EHSA analysis execution with logging
- Classification results and statistics
- Preprocessing function demonstrations

### 2. Requirements

Make sure PyEhsa is installed:
```bash
pip install -e ..  # If running from examples/ directory
# or
pip install pyehsa  # If installed from PyPI
```

### 3. Expected Output

The basic usage example will:
1. ✅ Create 128 synthetic observations (16 regions × 8 time periods)
2. 🔥 Run emerging hotspot analysis
3. 📊 Display classification distributions
4. 📈 Show Mann-Kendall trend statistics
5. 🔧 Demonstrate preprocessing functions

### 4. Understanding the Results

**Classifications you might see:**
- `no pattern detected`: Areas without significant spatial-temporal patterns
- `new hotspot`: Recently emerged hotspot areas
- `consecutive hotspot`: Areas that maintain hotspot status
- `oscillating hotspot`: Areas with intermittent hotspot behavior
- And more (see documentation for full list)

**Key Metrics:**
- **Tau values**: Mann-Kendall trend statistics (-1 to +1)
- **P-values**: Statistical significance of patterns
- **Classification confidence**: Based on spatial neighbors and temporal consistency

### 5. Next Steps

After running these examples, you can:
1. Modify the synthetic data parameters
2. Try with your own real datasets
3. Experiment with different EHSA parameters (`k`, `nsim`)
4. Visualize results using the plotting functions

## Support

For more information:
- 📖 Check the main README.md
- 🐛 Report issues on GitHub
- 💡 See docstrings in the code for detailed parameter explanations
