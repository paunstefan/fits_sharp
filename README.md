# FitsSharp

FitsSharp is a fast, GPU-accelerated tool that ranks astronomical FITS images by clarity. Perfect for astrophotographers who need to select the sharpest frames from large batches of telescope images. It focuses on Moon photos, not planetary.

## Features

- Ranks FITS images based on clarity metrics (Laplacian and Sobel)
- GPU acceleration with automatic CPU fallback

## Installation

```bash
cargo install --git https://github.com/paunstefan/fits_sharp
```

Or clone and build from source:

```bash
git clone https://github.com/paunstefan/fits_sharp
cd fits_sharp
cargo build --release
```

## Usage

```bash
# Basic usage
fits_sharp /path/to/images

# Show clarity scores
fits_sharp --verbose /path/to/images

# Force CPU processing
fits_sharp --cpu-only /path/to/images
```

## Example: Select Best Images

To select the top 100 sharpest images and copy them to another directory:

```bash
fits_sharp /path/to/images | head -n 100 | xargs -I{} cp {} /path/to/best_frames/
```

## Requirements

- Rust 1.60+
- `cfitsio` library
- FITS files with at least 2 dimensions
