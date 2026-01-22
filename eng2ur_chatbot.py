import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from datetime import datetime
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Urdu.AI | Premium Translator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN NEON CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&family=Noto+Nastaliq+Urdu&display=swap');

    /* Global Background */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Main Header */
    .main-header {
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Urdu Text Styling */
    .urdu-output {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.8rem !important;
        line-height: 2.0 !important;
        direction: rtl;
        text-align: right;
        color: #e2e8f0;
        padding: 20px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 15px;
        border-right: 5px solid #c084fc;
    }

    /* Custom Text Area */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.6) !important;
        color: white !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 15px !important;
        font-size: 1.1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 15px rgba(96, 165, 250, 0.2) !important;
    }

    /* Glowing Button */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        width: 100%;
        height: 50px;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4) !important;
    }

    /* Sidebar Fix */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION INITIALIZATION ---
if 'history' not in st.session_state: st.session_state.history = []
if 'model_ready' not in st.session_state: st.session_state.model_ready = False
if 'count' not in st.session_state: st.session_state.count = 0

# --- MODEL CORE ---
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    names = ["finetuned-opus", "Helsinki-NLP/opus-mt-en-ur"]
    for name in names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
            model.to("cpu")
            return tokenizer, model, name
        except: continue
    return None, None, None

# --- SIDEBAR UI ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/144/artificial-intelligence.png", width=80)
    st.markdown("## Urdu.AI Control")
    st.caption("v2.0 • Fine-tuned Neural Engine")
    
    if st.button("🔌 Connect AI Engine", type="primary"):
        with st.spinner("Initializing neural links..."):
            tok, mod, name = load_ai_engine()
            if mod:
                st.session_state.tokenizer = tok
                st.session_state.model = mod
                st.session_state.model_name = name
                st.session_state.model_ready = True
                st.toast("AI Engine Online!", icon="🚀")
    
    st.divider()
    st.markdown("### 📈 Session Stats")
    c1, c2 = st.columns(2)
    c1.metric("Translated", st.session_state.count)
    c2.metric("History", len(st.session_state.history))
    
    if st.button("🗑️ Reset Workspace"):
        st.session_state.history = []
        st.session_state.count = 0
        st.rerun()

# --- MAIN INTERFACE ---
st.markdown('<h1 class="main-header">Urdu.AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#94a3b8; margin-top:-10px;">The Next-Gen Translation Suite</p>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 English Source")
    src_text = st.text_area("", placeholder="Enter text to translate...", height=180, label_visibility="collapsed")
    translate_trigger = st.button("Translate Now ✨")

with col2:
    st.markdown("### 📥 Urdu Output")
    if translate_trigger and src_text.strip():
        if not st.session_state.model_ready:
            st.error("Please connect the AI Engine first from the sidebar!")
        else:
            with st.spinner("AI is thinking..."):
                start_time = time.time()
                # Inference Logic
                inputs = st.session_state.tokenizer(src_text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = st.session_state.model.generate(
                        **inputs, max_new_tokens=128, num_beams=5, early_stopping=True
                    )
                prediction = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
                end_time = time.time()
                
                # Render Output
                st.markdown(f'<div class="urdu-output">{prediction}</div>', unsafe_allow_html=True)
                
                # Save State
                st.session_state.count += 1
                st.session_state.history.append({
                    "time": datetime.now().strftime("%I:%M %p"),
                    "en": src_text,
                    "ur": prediction,
                    "lat": f"{end_time-start_time:.2f}s"
                })
                st.caption(f"Translated in {end_time-start_time:.2f}s")
    else:
        st.markdown('<div class="urdu-output" style="color:#475569; opacity:0.5; text-align:center;">ترجمہ یہاں ظاہر ہوگا</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- HISTORY SECTION ---
if st.session_state.history:
    st.markdown("### 📜 Recent Intelligence History")
    for idx, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Translation #{len(st.session_state.history)-idx} — {item['time']}"):
            hcol1, hcol2 = st.columns([1, 1])
            hcol1.markdown(f"**English:**\n{item['en']}")
            hcol2.markdown(f"<div style='direction:rtl; text-align:right; font-family:\"Noto Nastaliq Urdu\";'>{item['ur']}</div>", unsafe_allow_html=True)
            
            # Simple Copy Logic
            st.code(item['ur'], language=None)
            if st.button("🗑️ Delete", key=f"del_{idx}"):
                st.session_state.history.pop(-(idx+1))
                st.rerun()

# --- FOOTER ---
st.markdown(
    f"""<div style="text-align: center; color: #475569; padding: 2rem;">
        Model: <span style="color:#60a5fa">{st.session_state.get('model_name', 'Not Connected')}</span> | 
        Developed with Precision 🛠️
    </div>""", 
    unsafe_allow_html=True
)
