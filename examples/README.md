# FluxPipeline Examples

This directory contains example Jupyter notebooks and scripts demonstrating how to use FluxPipeline.

## 📓 Notebooks

### Getting Started
1. **[01_basic_generation.ipynb](01_basic_generation.ipynb)** - Basic image generation
   - Initialize the pipeline
   - Generate your first image
   - Understanding parameters

2. **[02_batch_processing.ipynb](02_batch_processing.ipynb)** - Batch image generation
   - Generate multiple images
   - Use different prompts
   - Batch processing tips

3. **[03_gif_creation.ipynb](03_gif_creation.ipynb)** - Create animated GIFs
   - Generate image sequences
   - Create smooth transitions
   - Export as GIF

4. **[04_custom_prompts.ipynb](04_custom_prompts.ipynb)** - Advanced prompt engineering
   - Prompt optimization techniques
   - Negative prompts
   - Style modifiers

### Advanced Topics
5. **[05_memory_optimization.ipynb](05_memory_optimization.ipynb)** - Memory management
   - Configure memory settings
   - GPU optimization
   - Handle OOM errors

6. **[06_seed_management.ipynb](06_seed_management.ipynb)** - Reproducible generation
   - Understand seed profiles
   - Reproducible results
   - Seed statistics

## 🚀 Quick Start

### Prerequisites
```bash
# Install Jupyter
pip install jupyter ipykernel

# Register the flux environment
python -m ipykernel install --user --name=flux

# Start Jupyter
jupyter notebook examples/
```

### Running Examples
1. Open Jupyter Notebook
2. Navigate to the `examples/` directory
3. Select a notebook
4. Run cells sequentially (Shift+Enter)

## 📚 Example Scripts

Python scripts are also provided for those who prefer non-interactive examples:

- `example_basic.py` - Basic generation script
- `example_batch.py` - Batch processing script
- `example_gif.py` - GIF creation script

Run with:
```bash
python examples/example_basic.py
```

## 💡 Tips

- Start with `01_basic_generation.ipynb` if you're new
- Each notebook is self-contained and can be run independently
- Modify parameters to experiment
- Check the main README.md for hardware requirements

## 🐛 Troubleshooting

### Kernel Not Found
If Jupyter can't find the `flux` kernel:
```bash
conda activate flux
python -m ipykernel install --user --name=flux --display-name "Python (flux)"
```

### CUDA Out of Memory
- Reduce image dimensions
- Enable attention slicing
- See notebook 05 for memory optimization

### Import Errors
Ensure you're in the flux environment and have installed all dependencies:
```bash
conda activate flux
pip install -r requirements.txt
```

## 📖 Additional Resources

- [Main Documentation](../README.md)
- [API Documentation](../docs/)
- [Troubleshooting Guide](../README.md#troubleshooting)

---

**Note**: These examples use the FLUX.1-schnell model which is not for commercial use.
Please review the [LICENSE](../LICENSE) before using these examples.
