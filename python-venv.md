# Python environment

Here are the main ways to create a Python environment:

### Using venv (built-in, simplest)

```
##bash
# Create
python -m venv myenv

# Activate
source myenv/bin/activate        # macOS/Linux
myenv\Scripts\activate           # Windows

# Deactivate
deactivate
```

---
### Using conda (popular for data science)

```
##bash
# Create
conda create -n myenv python=3.11

# Activate
conda activate myenv

# Deactivate
conda deactivate
```

---
### Using uv (modern, very fast)

```
## bash
# Install uv first
pip install uv

# Create and activate
uv venv myenv
source myenv/bin/activate        # macOS/Linux
myenv\Scripts\activate           # Windows
```

---
### Managing packages
Once your environment is active, install packages normally:

```
## bash
pip install genesis-world numpy torch
```
Save and restore dependencies:
```
## bash
pip freeze > requirements.txt       # save
pip install -r requirements.txt     # restore
```

### Which to choose?

* venv — good default, no extra install needed
* conda — best when you need non-Python dependencies (CUDA, C libs)
* uv — fastest option, great for new projects

For Genesis specifically, conda is often recommended since it handles PyTorch + CUDA dependencies cleanly.

