# ... [Keep all the existing imports and helper functions unchanged] ...

# Update the CSS section with modern design elements
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
        * { 
            font-family: 'Inter', sans-serif;
            box-sizing: border-box;
        }
        
        .main { 
            background: linear-gradient(152deg, #0a192f 0%, #172a45 100%);
            color: #ffffff;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #6366f1 0%, #a855f7 100%);
            color: white !important;
            border-radius: 12px;
            padding: 14px 28px;
            border: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 500;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(99, 102, 241, 0.3);
        }
        
        .glass-panel {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .metric-card {
            background: linear-gradient(45deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
        }
        
        .stFileUploader>div>div>div>div {
            color: #ffffff;
            border: 2px dashed #6366f1;
            background: rgba(99, 102, 241, 0.05);
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .stFileUploader>div>div>div>div:hover {
            border-color: #a855f7;
            background: rgba(168, 85, 247, 0.05);
        }
        
        .game-card {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 16px;
            overflow: hidden;
            position: relative;
        }
        
        .game-card:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }
        
        .progress-bar {
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            transition: width 0.5s ease;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.98); }
            50% { transform: scale(1.02); }
            100% { transform: scale(0.98); }
        }
    </style>
""", unsafe_allow_html=True)

# ... [Keep the load_model, get_image_hash, and predict_image functions unchanged] ...

# =======================
# Enhanced Main Page Design
# =======================
def main():
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=500)
    except Exception as e:
        st.write("Logo not found.")

    with st.container():
        st.markdown("""
        <div class="glass-panel pulse" style="text-align: center; margin-bottom: 2rem;">
            <h1 style="margin:0; color: #e0e7ff;">Deepfake Detection System</h1>
            <p style="opacity: 0.8;">Advanced AI-powered image authentication</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div class="glass-panel">
            <h2 style="color: #a855f7; margin-top:0;">🔮 Navigation</h2>
            <button onclick="window.streamlit.setComponentValue('main')" 
                    style="width: 100%; margin: 0.5rem 0; background: rgba(99, 102, 241, 0.1); border: 1px solid #6366f1; color: #e0e7ff; border-radius: 8px; padding: 12px;">
                🧪 Image Analysis
            </button>
            <button onclick="window.streamlit.setComponentValue('game')" 
                    style="width: 100%; margin: 0.5rem 0; background: rgba(168, 85, 247, 0.1); border: 1px solid #a855f7; color: #e0e7ff; border-radius: 8px; padding: 12px;">
                🎮 Detection Game
            </button>
        </div>
        """, unsafe_allow_html=True)

    # Rest of main page content remains structurally the same, but update elements with new classes:
    
    # Update columns to use glass-panel
    col1, col2 = st.columns([4, 3])
    with col1:
        st.markdown("""
        <div class="glass-panel">
            <h3 style="margin-top:0;">📤 Image Analysis Zone</h3>
        """, unsafe_allow_html=True)
        # ... [rest of column 1 content] ...
    
    with col2:
        st.markdown("""
        <div class="glass-panel">
            <h3 style="margin-top:0;">🔍 Preview</h3>
        """, unsafe_allow_html=True)
        # ... [rest of column 2 content] ...

    # Update results display with new metric cards
    st.markdown("""
    <div class="glass-panel">
        <h3 style="margin-top:0;">📊 Detection Report</h3>
    """, unsafe_allow_html=True)
    
    # Update charts with modern visualization
    col_chart_left, col_chart_right = st.columns(2)
    with col_chart_left:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin:0; color: #00ff88;">Real Confidence</h4>
            <h1 style="margin:0;">{score}%</h1>
        </div>
        """.format(score=scores.get("real", 0)*100), unsafe_allow_html=True)
    
    # ... [rest of main page updates] ...

# =======================
# Enhanced Game Page Design
# =======================
def game():
    st.markdown("""
    <div class="glass-panel" style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin:0; color: #e0e7ff;">Deepfake Detective Challenge</h1>
        <p style="opacity: 0.8;">Identify authentic content in 5 rounds</p>
    </div>
    """, unsafe_allow_html=True)

    # Add animated progress bar
    st.markdown(f"""
    <div style="margin: 2rem 0;">
        <div class="progress-bar">
            <div class="progress-fill" style="width: {(st.session_state.game_round/5)*100}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Update game cards with hover effects
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="game-card">
            {image}
        </div>
        """.format(image=left_fixed), unsafe_allow_html=True)
    
    # ... [rest of game page updates] ...

# ... [Keep the page routing and remaining code unchanged] ...
