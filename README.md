# <a name="aura-vision-ai-image-generator"></a>🔮 Aura Vision – AI Image Generator
A full-featured AI-powered image generation web app built with **Streamlit** and **Stable Diffusion XL**, offering both **Text-to-Image** and **Image Style Transfer** modes. Designed with a modern UI, GPU memory optimizations, and customizable visual effects.

-----
## <a name="features"></a>🚀 Features
- ✨ **Text-to-Image generation** using Stable Diffusion XL
- 🎨 **Style Transfer** (e.g., Anime, Ghibli, etc.) using Img2Img pipeline
- 🧠 **CLIP-based prompt understanding**
- 🗞️ **Negative prompt support** for output refinement
- 🎛️ **Post-processing effects**: blur, sharpen, grayscale, etc.
- 🌈 **Custom UI themes**: Cyberpunk, Sunset Gradient, Light
- 📏 Image resizing, cropping, and **metadata display**
- 💾 **Download final images** with custom filenames
- ⚡ **GPU memory cleanup** for performance boost
- 📊 Real-time **progress bar** during generation
- 🧠 Built with PyTorch, HuggingFace Diffusers, and PIL
-----
## <a name="tech-stack"></a>🛠️ Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **AI Models**: Stable Diffusion XL, CLIP, VAE, U-Net
- **Libraries**: diffusers, torch, Pillow, numpy
-----

## <a name="how-to-run"></a>🧑‍💻 How to Run
*# Clone the repository*
git clone https://github.com/GauravisSerious/AuraVision
cd aura-vision

*# (Optional) Create a virtual environment*
python -m venv venv
source venv/bin/activate  *# or venv\Scripts\activate on Windows*

*# Install dependencies*
pip install -r requirements.txt

*# Run the app*
streamlit run app.py

-----
## <a name="model-requirements"></a>🧠 Model Requirements
- HuggingFace access token to download Stable Diffusion XL weights
- CUDA-compatible GPU recommended for fast generation
----

