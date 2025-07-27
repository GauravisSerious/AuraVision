import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import time
import re
import io
import random

# Set Streamlit page configuration
st.set_page_config(
    page_title="Aura Vision - AI Image Generator",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme CSS definitions
def get_theme_css(theme_name):
    themes = {
        "Light": """
            .stApp { 
                background: #ffffff !important; 
                color: #333333 !important; 
            }
            h1, h2, h3 { 
                background: linear-gradient(90deg, #3a7bd5, #00d2ff) !important; 
                -webkit-background-clip: text !important; 
            }
            .stButton > button { 
                background: linear-gradient(90deg, #3a7bd5, #00d2ff) !important;
            }
            .css-card {
                background-color: #f8f9fa !important;
                color: #333333 !important;
                border: 1px solid #eaeaea !important;
            }
            .css-sidebar {
                background-color: #f0f2f6 !important;
                color: #333333 !important;
                border: 1px solid #eaeaea !important;
            }
            .streamlit-expanderHeader {
                background-color: #f0f2f6 !important;
            }
            .stTextInput > div > div > input {
                background-color: #ffffff !important;
                color: #333333 !important;
                border: 1px solid #dddddd !important;
            }
            .stSelectbox > div > div {
                background-color: #ffffff !important;
                color: #333333 !important;
                border: 1px solid #dddddd !important;
            }
            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, #3a7bd5, #00d2ff) !important;
            }
        """,
        "Sunset Gradient": """
            .stApp { background: linear-gradient(to right, #121212, #1E1E1E) !important; color: #FFFFFF !important; }
            h1, h2, h3 { background: linear-gradient(90deg, #FF9966, #FF5E62) !important; -webkit-background-clip: text !important; }
            .stButton > button { background: linear-gradient(90deg, #FF9966, #FF5E62) !important; }
            .css-card { background-color: rgba(30, 34, 40, 0.8) !important; color: #FFFFFF !important; }
            .css-sidebar { background-color: rgba(25, 28, 36, 0.8) !important; color: #FFFFFF !important; }
        """,
        "Cyberpunk": """
            .stApp { background: linear-gradient(to right, #0a0a0a, #121212) !important; color: #FFFFFF !important; }
            h1, h2, h3 { background: linear-gradient(90deg, #0ff0fc, #7b24d3) !important; -webkit-background-clip: text !important; }
            .stButton > button { background: linear-gradient(90deg, #0ff0fc, #7b24d3) !important; }
            .css-card { background-color: rgba(15, 18, 23, 0.8) !important; color: #FFFFFF !important; }
            .css-sidebar { background-color: rgba(15, 18, 23, 0.8) !important; color: #FFFFFF !important; }
        """,
        "Minimalist": """
            .stApp { background: #f5f5f5 !important; color: #222222 !important; }
            h1, h2, h3 { background: linear-gradient(90deg, #222222, #555555) !important; -webkit-background-clip: text !important; }
            .stButton > button { background: linear-gradient(90deg, #222222, #444444) !important; color: white !important; }
            .css-card, .css-sidebar { background-color: rgba(255, 255, 255, 0.8) !important; color: #222222 !important; }
        """,
        "Ocean": """
            .stApp { background: linear-gradient(to right, #141E30, #243B55) !important; color: #FFFFFF !important; }
            h1, h2, h3 { background: linear-gradient(90deg, #56CCF2, #2F80ED) !important; -webkit-background-clip: text !important; }
            .stButton > button { background: linear-gradient(90deg, #56CCF2, #2F80ED) !important; }
            .css-card { background-color: rgba(20, 36, 64, 0.8) !important; color: #FFFFFF !important; }
            .css-sidebar { background-color: rgba(20, 36, 64, 0.8) !important; color: #FFFFFF !important; }
        """,
        "Forest": """
            .stApp { background: linear-gradient(to right, #134E5E, #1a1a1a) !important; color: #FFFFFF !important; }
            h1, h2, h3 { background: linear-gradient(90deg, #71B280, #134E5E) !important; -webkit-background-clip: text !important; }
            .stButton > button { background: linear-gradient(90deg, #71B280, #134E5E) !important; }
            .css-card { background-color: rgba(19, 45, 41, 0.8) !important; color: #FFFFFF !important; }
            .css-sidebar { background-color: rgba(19, 45, 41, 0.8) !important; color: #FFFFFF !important; }
        """,
        "Dark": """
            .stApp { background: #0E1117 !important; color: #FFFFFF !important; }
            h1, h2, h3 { background: linear-gradient(90deg, #BBB, #EEE) !important; -webkit-background-clip: text !important; }
            .stButton > button { background: linear-gradient(90deg, #444, #777) !important; }
            .css-card { background-color: rgba(30, 34, 40, 0.8) !important; color: #FFFFFF !important; }
            .css-sidebar { background-color: rgba(25, 28, 36, 0.8) !important; color: #FFFFFF !important; }
        """
    }
    
    return themes.get(theme_name, themes["Dark"])

# Custom CSS for base styling
def load_base_css():
    st.markdown("""
    <style>
    /* Base styling that will be consistent across themes */
    
    /* Card styling */
    .css-card {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .css-sidebar {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding-bottom: 10px;
        margin-bottom: 20px;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Logo */
    .logo {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .logo-text {
        font-size: 100px;
        font-weight: bold;
        margin-left: 10px;
        -webkit-background-clip: text;
        //-webkit-text-fill-color: transparent;
    }
    
    /* Title styling */
    .title-text {
        font-family: 'Audiowide', 'Orbitron', 'Segoe UI', sans-serif;
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-image: linear-gradient(90deg, #fc00ff, #00dbde);
        text-shadow: 0 0 15px rgba(252, 0, 255, 0.5), 0 0 25px rgba(0, 219, 222, 0.3);
        margin-bottom: 15px;
        padding-bottom: 10px;
    }
    
    .subtitle-text {
        font-family: 'Quicksand', 'Segoe UI', sans-serif;
        font-size: 18px;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.8);
        letter-spacing: 1px;
        margin-bottom: 30px;
    }
    
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Audiowide&family=Orbitron:wght@400;700&family=Quicksand:wght@300;400;500&display=swap');
    
    /* Image display area */
    .image-display {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
        opacity: 0.7;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply theme CSS
def apply_theme(theme_name):
    theme_css = get_theme_css(theme_name)
    st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

# Initialize the image generation pipeline
@st.cache_resource
def initialize_image_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    return pipe, device

# Generate image with callback for progress bar
def generate_image(image_pipe, prompt, negative_prompt="", steps=30, guidance_scale=7.5):
    progress_bar = st.progress(0)
    progress_status = st.empty()
    
    # Callback function to update progress bar
    def callback_fn(step, timestep, latents):
        progress = (step + 1) / steps
        progress_bar.progress(progress)
        progress_status.markdown(f"<div class='loading-pulse'>Generating image... Step {step+1}/{steps} ({int(progress*100)}%)</div>", unsafe_allow_html=True)
        return None
    
    image = image_pipe(
        prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=steps, 
        guidance_scale=guidance_scale,
        callback=callback_fn,
        callback_steps=1
    ).images[0]
    
    progress_status.empty()
    return image

# Enhance prompt function
def enhance_prompt(prompt):
    suggestions = []
    
    # Check prompt length
    if len(prompt.split()) < 3:
        suggestions.append("Add more details to make your prompt more descriptive")
    
    # Check for style keywords
    style_keywords = ["oil painting", "digital art", "watercolor", "3D render", "photography", "sketch"]
    if not any(style in prompt.lower() for style in style_keywords):
        suggestions.append("Consider adding an art style (e.g., 'oil painting', 'digital art', 'photography')")
    
    # Check for lighting/mood keywords
    lighting_keywords = ["sunlight", "dark", "night", "bright", "moody", "dramatic lighting"]
    if not any(light in prompt.lower() for light in lighting_keywords):
        suggestions.append("Add lighting description (e.g., 'dramatic lighting', 'soft sunlight')")
    
    # Check for composition keywords
    composition_keywords = ["close-up", "wide angle", "portrait", "landscape", "aerial view"]
    if not any(comp in prompt.lower() for comp in composition_keywords):
        suggestions.append("Specify composition (e.g., 'close-up', 'wide angle')")
    
    return suggestions

# Apply effects to the image
def apply_effects(image, effect):
    if effect == "Blur":
        return image.filter(ImageFilter.BLUR)
    elif effect == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif effect == "Grayscale":
        return image.convert("L")
    elif effect == "Sepia":
        sepia_filter = Image.new("RGB", image.size, (255, 240, 192))
        return Image.blend(image.convert("RGB"), sepia_filter, 0.5)
    elif effect == "Vintage":
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(0.5)
    elif effect == "Warm":
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(1.5)
    elif effect == "Cool":
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(0.7)
    elif effect == "Black & White":
        return image.convert("1")
    elif effect == "Edge Detection":
        return image.filter(ImageFilter.FIND_EDGES)
    elif effect == "Emboss":
        return image.filter(ImageFilter.EMBOSS)
    elif effect == "Color Inversion":
        return ImageOps.invert(image)
    elif effect == "Oil Painting":
        return image.filter(ImageFilter.SMOOTH_MORE)
    elif effect == "Sketch":
        grayscale = image.convert("L")
        inverted = ImageOps.invert(grayscale)
        return ImageOps.colorize(inverted, black="black", white="white")
    elif effect == "Pixelation":
        small = image.resize((32, 32), resample=Image.NEAREST)
        return small.resize(image.size, Image.NEAREST)
    else:
        return image

# Main function
def main():
    # Load base CSS
    load_base_css()
    
    # Initialize session state for storing the generated image and settings
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'prompt' not in st.session_state:
        st.session_state.prompt = ""
    if 'negative_prompt' not in st.session_state:
        st.session_state.negative_prompt = ""
    if 'effect' not in st.session_state:
        st.session_state.effect = "None"
    if 'generated_time' not in st.session_state:
        st.session_state.generated_time = ""
    
    # Sidebar with card styling
    with st.sidebar:
        st.markdown('<div class="css-sidebar">', unsafe_allow_html=True)
        st.markdown('### 🎨 UI Theme')
        theme_options = ["Light", "Sunset Gradient", "Cyberpunk", "Minimalist", "Ocean", "Forest", "Dark"]
        selected_theme = st.selectbox("Select a theme", theme_options, index=2)
        
        # Apply selected theme
        apply_theme(selected_theme)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="css-sidebar">', unsafe_allow_html=True)
        st.markdown('### 🛠️ Generation Settings')
        steps = st.slider("Number of steps", min_value=10, max_value=100, value=30)
        guidance_scale = st.slider("Guidance scale", min_value=1.0, max_value=20.0, value=7.5)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Image effects section
        st.markdown('<div class="css-sidebar">', unsafe_allow_html=True)
        st.markdown('### 🎭 Image Effects')
        effect_options = ["None", "Blur", "Sharpen", "Grayscale", "Sepia", "Vintage", 
                          "Warm", "Cool", "Black & White", "Edge Detection", "Emboss", 
                          "Color Inversion", "Oil Painting", "Sketch", "Pixelation"]
        
        # Simple approach: Just get the selected effect
        selected_effect = st.selectbox("Choose an effect", effect_options)
        
        # Apply the effect immediately if an image exists and the effect has changed
        if st.session_state.original_image is not None and selected_effect != st.session_state.effect:
            st.session_state.effect = selected_effect
            st.session_state.image = apply_effects(st.session_state.original_image, selected_effect)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="css-sidebar">', unsafe_allow_html=True)
        st.markdown('### 📐 Image Dimensions')
        resize_option = st.checkbox("Resize Image")
        if resize_option:
            col1, col2 = st.columns(2)
            with col1:
                width = st.number_input("Width", min_value=100, max_value=2000, value=512)
            with col2:
                height = st.number_input("Height", min_value=100, max_value=2000, value=512)
        
        crop_option = st.checkbox("Crop Image")
        if crop_option:
            aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "4:3"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Logo and title
    st.markdown('<div class="logo"><span class="logo-text">🗿</span></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="title-text" style="text-align: center;">AURA VISION Image Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text" style="text-align: center;">Create stunning AI art with Stable Diffusion XL</p>', unsafe_allow_html=True)
    
    # Main content
    main_container = st.container()
    
    with main_container:
        # Prompt input section
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('### ✨ Create Your Image')
        
        # Prompt templates section
        prompt_templates = {
            "None (Custom)": "",
            "Fantasy Landscape": "A magical fantasy landscape with mountains, waterfalls, and a castle, digital art, trending on artstation",
            "Portrait": "Professional portrait photograph of a person, detailed features, studio lighting, high quality, 85mm lens",
            "Sci-Fi City": "Futuristic cyberpunk cityscape with neon lights, flying cars, and tall skyscrapers, at night, trending on artstation",
            "Nature": "Beautiful natural landscape with mountains, trees, and a lake, golden hour lighting, photorealistic",
            "Abstract Art": "Vibrant abstract digital art with fluid shapes and dynamic colors, trending on artstation",
            "Food": "Professional food photography of a gourmet dish, studio lighting, high resolution, shallow depth of field"
        }
        
        col1, col2 = st.columns([1, 3])
        with col1:
            template_choice = st.selectbox("Template:", list(prompt_templates.keys()))
        
        with col2:
            if template_choice != "None (Custom)":
                prompt = st.text_input("Customize:", value=prompt_templates[template_choice], placeholder="Describe your image here...")
            else:
                prompt = st.text_input("Your prompt:", placeholder="Describe your image here...")
        
        # Prompt enhancement
        if prompt:
            suggestions = enhance_prompt(prompt)
            if suggestions:
                with st.expander("💡 Prompt enhancement suggestions"):
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
        
        # Negative prompt
        negative_prompt = st.text_input("Negative prompt:", placeholder="Things to avoid in the image (e.g., blurry, bad quality, distorted)")
        
        # Generation button
        generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
        with generate_col2:
            generate_button = st.button("🚀 Generate Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Image generation process
        if generate_button:
            if not prompt:
                st.error("Please enter a prompt!")
            else:
                try:
                    # Initialize image pipeline
                    image_pipe, device = initialize_image_pipeline()
                    
                    with st.spinner():
                        # Generate image with progress bar
                        image = generate_image(image_pipe, prompt, negative_prompt, steps, guidance_scale)
                    
                        # Store the original image before any effects
                        st.session_state.original_image = image.copy()
                    
                        # Apply effects
                        if selected_effect != "None":
                            image = apply_effects(image, selected_effect)
                    
                        # Resize image
                        if resize_option:
                            image = image.resize((width, height))
                    
                        # Crop image
                        if crop_option:
                            img_width, img_height = image.size
                            if aspect_ratio == "1:1":
                                size = min(img_width, img_height)
                                image = image.crop(((img_width - size) // 2, (img_height - size) // 2, 
                                                (img_width + size) // 2, (img_height + size) // 2))
                            elif aspect_ratio == "16:9":
                                new_height = img_width * 9 // 16
                                image = image.crop((0, (img_height - new_height) // 2, 
                                                img_width, (img_height + new_height) // 2))
                            elif aspect_ratio == "4:3":
                                new_height = img_width * 3 // 4
                                image = image.crop((0, (img_height - new_height) // 2, 
                                                img_width, (img_height + new_height) // 2))
                    
                    # Store generated image and settings in session state
                    st.session_state.image = image
                    st.session_state.prompt = prompt
                    st.session_state.negative_prompt = negative_prompt
                    st.session_state.effect = selected_effect
                    st.session_state.generated_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Success animation
                    st.balloons()
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        # Display image section (always show if image exists in session state)
        if st.session_state.image is not None:
            image = st.session_state.image
            prompt = st.session_state.prompt
            effect = st.session_state.effect
            
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Your Generated Image")
            st.markdown('<div class="image-display">', unsafe_allow_html=True)
            
            # Image display size control
            display_options = st.columns([3, 1])
            with display_options[1]:
                display_size = st.select_slider(
                    "Display size",
                    options=["Small", "Medium", "Large", "Original"],
                    value="Medium"
                )
            
            # Determine the width based on selected display size
            if display_size == "Small":
                display_width = 400
            elif display_size == "Medium":
                display_width = 600
            elif display_size == "Large":
                display_width = 800
            else:  # Original
                display_width = None
                
            # Display the image with controlled width
            st.image(image, caption=f"'{prompt}' with {effect} effect", width=display_width)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download section
            col1, col2 = st.columns(2)
            
            # Prepare image for download
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            
            with col1:
                st.download_button(
                    label="📥 Download Image",
                    data=buffer,
                    file_name="cursio_image.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Option to download with custom filename
            with col2:
                filename = prompt.lower().replace(" ", "_")[:30]
                st.download_button(
                    label="📥 Download with Custom Name",
                    data=buffer,
                    file_name=f"{filename}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Simple button to reset to original image
            if effect != "None" and st.session_state.original_image is not None:
                if st.button("↩️ Reset to Original Image", use_container_width=True):
                    # Just set the image back to original directly
                    st.session_state.effect = "None"
                    st.session_state.image = st.session_state.original_image.copy()
            
            # Image metadata
            with st.expander("🔍 Image Details"):
                metadata_col1, metadata_col2 = st.columns(2)
                with metadata_col1:
                    st.write("**Dimensions:**", f"{image.width} x {image.height}")
                    st.write("**Steps:**", steps)
                    st.write("**Guidance Scale:**", guidance_scale)
                with metadata_col2:
                    st.write("**Effect Applied:**", effect)
                    st.write("**Generated On:**", st.session_state.generated_time)
            
            # Success message with animation
            st.markdown('<div style="text-align: center; margin-top: 10px;">', unsafe_allow_html=True)
            st.success("🎉 Image generated successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    # st.markdown('<div class="footer">', unsafe_allow_html=True)
    # st.markdown("AV Image Generator © 2025 | Powered by Stable Diffusion XL")
    # st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()