import streamlit as st
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import text
import sqlparse
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import folium
from streamlit_globe import streamlit_globe
from streamlit_folium import st_folium
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
DB_NAME = st.secrets.get("DB_NAME", os.getenv("DB_NAME"))
DB_USER = st.secrets.get("DB_USER", os.getenv("DB_USER"))
DB_PASS = st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD"))
DB_HOST = st.secrets.get("DB_HOST", os.getenv("DB_HOST"))
DB_PORT = st.secrets.get("DB_PORT", os.getenv("DB_PORT"))
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

# Configure DB engine
engine = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

st.set_page_config(
    page_title="FloatChat AI - ARGO Ocean Data Explorer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
model1 = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

@st.cache_data(show_spinner=False)
def load_data():
    # Example: reading all or partial data from DB
    query = "SELECT * FROM argo_profiles LIMIT 100000"
    df = pd.read_sql(query, engine)
    return df

if "df" not in st.session_state:
    st.session_state.df = load_data()
df = st.session_state.df


# Prompt prefix for SQL LLM
PROMPT_PREFIX = """
    You are an expert in converting English questions to SQL code for a PostgreSQL + PostGIS database.!  
    The SQL database has the name argo_profiles and has the following columns:  
    platform_number, cycle_number, juld (DATE), latitude, longitude, pres_adjusted, temp_adjusted, psal_adjusted, pres_adjusted_qc, temp_adjusted_qc, psal_adjusted_qc.

    For example,  
    Example 1 - What is the maximum temperature in the Indian Ocean?  
    SELECT MAX(temp_adjusted) FROM argo_profiles WHERE latitude BETWEEN -60 AND 30 AND longitude BETWEEN 20 AND 120;

    Example 2 - What is the average salinity in the Arabian Sea in 2020?  
    SELECT AVG(psal_adjusted) FROM argo_profiles WHERE latitude BETWEEN 5 AND 25 AND longitude BETWEEN 50 AND 90 AND EXTRACT(YEAR FROM juld) = 2020;

    Example 3 - Show all profiles where temp_adjusted_qc = '1'.  
    SELECT * FROM argo_profiles WHERE temp_adjusted_qc = '1';

    Example 4 - List temperature and salinity for platform 7689.  
    SELECT temp_adjusted, psal_adjusted FROM argo_profiles WHERE platform_number = 7689;

    Example 5 - Give the temperature trend in the Bay of Bengal for the last 5 years.  
    SELECT EXTRACT(YEAR FROM juld) AS year, AVG(temp_adjusted) AS avg_temp FROM argo_profiles WHERE latitude BETWEEN 5 AND 25 AND longitude BETWEEN 80 AND 100 AND juld >= (CURRENT_DATE - INTERVAL '5 years') GROUP BY year ORDER BY year;

    Example 6 - Find the 5 closest profiles to coordinates 12.200,94.719.  
    SELECT platform_number, latitude, longitude  FROM argo_profiles a WHERE platform_number IN (SELECT DISTINCT platform_number FROM argo_profiles) ORDER BY SQRT(POWER(latitude - 12.200, 2) + POWER(longitude - 94.719, 2)) LIMIT 5;     
    
    Example 7 - Find the average temperature and salinity for deepest 20% pressures per platform  
    WITH ranked AS ( SELECT *, PERCENT_RANK() OVER (PARTITION BY platform_number ORDER BY pres_adjusted DESC) AS pr FROM argo_profiles ) SELECT platform_number, AVG(temp_adjusted) AS avg_temp, AVG(psal_adjusted) AS avg_salinity FROM ranked WHERE pr >= 0.8 GROUP BY platform_number ORDER BY platform_number DESC;

    Example 8 - Find a float with temperature between 25 and 30 and psal between 10 and 20
    SELECT DISTINCT platform_number FROM argo_profiles WHERE temp_adjusted BETWEEN 25 AND 30 AND psal_adjusted BETWEEN 10 AND 20;
    
    Dont include ```
    """
# Styling and ocean header
def render_ocean_header():
    ocean_html = """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 50px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
                        radial-gradient(circle at 70% 80%, rgba(255,255,255,0.1) 0%, transparent 50%);
        "></div>
        <h1 style="
            color: white;
            font-size: 4em;
            margin: 0;
            text-shadow: 2px 2px 15px rgba(0,0,0,0.4);
            font-family: 'Segoe UI', sans-serif;
            font-weight: 800;
            position: relative;
            z-index: 2;
            letter-spacing: -2px;
        ">🌊 FloatChat AI</h1>
        <p style="
            color: rgba(255,255,255,0.95);
            font-size: 1.4em;
            margin: 20px 0 0 0;
            position: relative;
            z-index: 2;
            font-weight: 300;
            letter-spacing: 1px;
        ">Advanced Ocean Data Analytics Platform</p>
    </div>
    """
    st.markdown(ocean_html, unsafe_allow_html=True)

st.set_page_config(page_title="FloatChat AI", layout="wide", initial_sidebar_state="collapsed")
render_ocean_header()

# CSS Styles
st.markdown("""
    <style>
        /* Hide default Streamlit elements */
        #MainMenu, footer, header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Main container styling */
        .main .block-container {
            padding: 0rem 1rem 1rem 1rem;
            max-width: 1400px;
        }
        
        /* Body and background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #24243e, #313264);
            min-height: 100vh;
        }
        
        /* Enhanced button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102,126,234,0.3);
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102,126,234,0.4);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        .stButton > button:active {
            transform: translateY(0px);
        }

        /* Chat container fixes */
        .chat-container {
            background: transparent;
            padding: 0;
            margin: 0;
            border: none;
        }
        
        /* Enhanced chat messages */
        .chat-message {
            margin: 15px 0;
            display: flex;
            align-items: flex-start;
            animation: slideIn 0.5s ease-out;
        }
        
        .message-bubble {
            max-width: 75%;
            padding: 18px 24px;
            border-radius: 20px;
            font-size: 15px;
            line-height: 1.6;
            word-wrap: break-word;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .user-message {
            justify-content: flex-end;
        }
        .assistant-message {
            justify-content: flex-start;
        }

        .user-bubble {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-bottom-right-radius: 8px;
        }
        .assistant-bubble {
            background: linear-gradient(135deg, rgba(67, 206, 162, 0.9), rgba(24, 90, 157, 0.9));
            color: white;
            border-bottom-left-radius: 8px;
        }

        /* Enhanced avatars */
        .message-avatar {
            width: 45px;
            height: 45px;
            border-radius: 50%;
            margin: 0 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            border: 2px solid rgba(255,255,255,0.2);
        }
        .user-avatar {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .assistant-avatar {
            background: linear-gradient(135deg, #43cea2, #185a9d);
            color: white;
        }

        /* Fixed chat input styling */
        .chat-input-container {
            bottom: 0;
            background: rgba(15, 12, 41, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 20px;
            margin: 20px 0 0 0;
            box-shadow: 0 -5px 25px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            z-index: 100;
        }
        
        /* Enhanced input field */
        .stTextInput > div > div > input {
            border-radius: 25px;
            border: 2px solid rgba(255,255,255,0.2);
            padding: 15px 25px;
            font-size: 16px;
            background: rgba(255,255,255,0.1);
            color: white;
            transition: all 0.3s ease;
        }
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.3);
            background: rgba(255,255,255,0.15);
        }
        .stTextInput > div > div > input::placeholder {
            color: rgba(255,255,255,0.7);
        }

        /* Enhanced metric cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            backdrop-filter: blur(15px);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            margin: 15px 0;
            color: white;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        }
        .metric-card h3 {
            font-size: 2.8em;
            margin: 0;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-card p {
            font-size: 1.2em;
            margin: 12px 0 0 0;
            font-weight: 500;
            opacity: 0.9;
        }

        /* Enhanced dataframe styling */
        .stDataFrame {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }

        /* Success/Info/Warning message styling */
        .stSuccess, .stInfo, .stWarning {
            border-radius: 15px;
            border: none;
            backdrop-filter: blur(10px);
        }

        /* Enhanced animations */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Remove white background from form */
        .stForm {
            background: transparent !important;
            border: none !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            color: rgba(255,255,255,0.8);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Helper functions for safe SQL validation, execution, AI generate, summary ...

def validate_sql_syntax(sql: str) -> bool:
    try:
        parsed = sqlparse.parse(sql)
        return len(parsed) > 0
    except Exception:
        return False

def is_safe_query(sql: str) -> bool:
    sql_upper = sql.strip().upper()
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT"]
    return sql_upper.startswith("SELECT") and all(x not in sql_upper for x in forbidden)

def read_sql_query(sql):
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        columns = result.keys()
    return [dict(zip(columns, row)) for row in rows]

def safe_execute(sql):
    if not validate_sql_syntax(sql):
        raise ValueError("Invalid SQL syntax generated.")
    if not is_safe_query(sql):
        raise ValueError("Unsafe SQL detected (non-SELECT or dangerous keyword).")
    return read_sql_query(sql)

def generate_gemini_response(question, prompt_prefix):
    max_retries = 2
    prompt_parts = [prompt_prefix, question]
    attempt = 0
    while attempt <= max_retries:
        response = model.generate_content(prompt_parts)
        sql = response.text.strip()
        try:
            output = safe_execute(sql)
            return {"sql": sql, "results": output}
        except ValueError as e:
            prompt_parts.append(f"⚠️ Invalid SQL: {e}. Please fix it and try again.")
            attempt += 1
    return {"error": "Failed to generate valid SQL."}

def generate_summary_llm(question, sql, results):
    if not results or "error" in results:
        return "No relevant data found for your question."
    df = pd.DataFrame(results)
    table_str = df.to_markdown(index=False)
    prompt = f"""
    You are a helpful assistant for ocean data analysis.
    Summarize the following SQL results for the user question: "{question}"
    SQL: {sql}
    Results:
    {table_str}
    Respond in 4 concise bullet points focusing on data.
    """
    summary_response = model1.generate_content(prompt)
    return summary_response.text.strip()

def query_backend(question):
    response = generate_gemini_response(question, PROMPT_PREFIX)
    sql = response.get("sql", "")
    results = response.get("results", [])
    summary = generate_summary_llm(question, sql, results) if "error" not in response else ""
    return {
        "sql": sql,
        "results": results,
        "summary": summary,
        "error": response.get("error", ""),
    }


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I'm your AI ocean data assistant. Ask me about temperatures, salinity, pressure patterns, or specific oceanic regions!",
        "type": "text"
    }]
if "current_view" not in st.session_state:
    st.session_state.current_view = 'chat'
if "platforms" not in st.session_state:
    st.session_state.platforms = []

# Navigation Buttons
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("💬 AI Chat", key="nav_chat", use_container_width=True):
        st.session_state.current_view = 'chat'
with col2:
    if st.button("📊 Data View", key="nav_raw", use_container_width=True):
        st.session_state.current_view = 'raw'
with col3:
    if st.button("📈 Analytics", key="nav_visualize", use_container_width=True):
        st.session_state.current_view = 'visualize'
with col4:
    if st.button("💾 Export", key="nav_download", use_container_width=True):
        st.session_state.current_view = 'download'
with col5:
    if st.button("🌍 Regional", key="nav_filter", use_container_width=True):
        st.session_state.current_view = 'filter'
st.markdown("---")

# UI Logic for each view:
if st.session_state.current_view == 'chat':
    st.markdown("## 💬 Intelligent Ocean Data Assistant")
    
    if not st.session_state.messages:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Hello! I'm your AI ocean data assistant. Ask me about temperatures, salinity, pressure patterns, or specific oceanic regions!", 
            "type": "text"
        }]
    
    # Chat display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-bubble user-bubble">
                    {message['content']}
                </div>
                <div class="message-avatar user-avatar">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            content = message['content']
            
            # Clean SQL formatting
            if "**SQL:**" in content:
                parts = content.split('\n\n')
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-avatar assistant-avatar">🤖</div>
                    <div class="message-bubble assistant-bubble">
                """, unsafe_allow_html=True)
                
                for part in parts:
                    if part.startswith("**SQL:**"):
                        sql_part = part.replace("**SQL:**", "").strip()
                        st.code(sql_part, language='sql')
                    else:
                        st.markdown(part)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-avatar assistant-avatar">🤖</div>
                    <div class="message-bubble assistant-bubble">
                        {content}
                """, unsafe_allow_html=True)
            
            if message.get("type") == "dataframe" and "dataframe" in message:
                st.dataframe(message["dataframe"], use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input section - ONLY in chat view
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask about ocean data...", 
                placeholder="e.g., What's the temperature trend in the Indian Ocean?",
                label_visibility="collapsed",
                key="user_query_input"
            )
        
        with col2:
            send_button = st.form_submit_button("Send 🚀", use_container_width=True)
    
    if send_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})
        
        with st.spinner("Analyzing your query..."):
            response = query_backend(user_input)
        
        error = response.get("error")
        sql = response.get("sql", "")
        summary = response.get("summary", "")
        results = response.get("results", [])
        
        if not error:
            df_result = pd.DataFrame(results) if results else pd.DataFrame()

            formatted_response = f"""
{sql}
                </pre>

**Summary:**
                    {summary.replace('- ', '• ')}
               
            """

            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted_response,
                "type": "html"
            })

            if not df_result.empty:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "📊 <b>Data Results:</b>",
                    "type": "dataframe",
                    "dataframe": df_result
                })

                st.rerun()

# Additional CSS to prevent duplication
    st.markdown("""
<style>
/* Ensure only one chat input container */
.chat-input-container:not(:first-of-type) {
    display: none !important;
}

/* Hide any duplicate text inputs */
.stTextInput:has(input[data-testid="textinput"]) ~ .stTextInput {
    display: none !important;
}

/* Ensure form cleanup */
.stForm[data-testid="form"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Fix for multiple form elements */
[data-testid="stForm"]:not(:first-child) {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
elif st.session_state.current_view == 'visualize':
    st.markdown("## 📈 Advanced Ocean Data Visualizations")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Geographic Maps", "📊 Statistical Charts", "🌍 3D Globe", "📈 Advanced Analytics", "🔬 Deep Analysis"])

    with tab1:
        st.subheader("🌍 Global Ocean Float Distribution")
        if 'latitude' in df.columns and 'longitude' in df.columns:
            sample_df = df.sample(min(2000, len(df)))
            cols_to_keep = ["latitude", "longitude"]
            if "temp_adjusted" in sample_df.columns:
                cols_to_keep.append("temp_adjusted")
            if "psal_adjusted" in sample_df.columns:
                cols_to_keep.append("psal_adjusted")

            map_df = sample_df[cols_to_keep].copy()
            map_df = map_df.astype(float, errors="ignore")  # force numeric cols to proper floats
            map_df = map_df.reset_index(drop=True)

            col1, col2 = st.columns(2)
            with col1:                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position=["longitude", "latitude"],
                    get_fill_color=[0, 128, 255, 160],
                    get_radius=40000,  
                    pickable=True
                )

                tooltip_html = "<b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}"

                if "temp_adjusted" in sample_df.columns:
                    tooltip_html += "<br/><b>Temp:</b> {temp_adjusted}"
                if "psal_adjusted" in sample_df.columns:
                    tooltip_html += "<br/><b>Salinity:</b> {psal_adjusted}"

                tooltip = {
                    "html": tooltip_html,
                    "style": {"backgroundColor": "steelblue", "color": "white"}
}

                view_state = pdk.ViewState(
                    latitude=float(sample_df["latitude"].mean()),
                    longitude=float(sample_df["longitude"].mean()),
                    zoom=2,
                    pitch=0
                )

                # Deck instance
                r = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style="light"  # "dark", "satellite"
                )

                try:
                    st.pydeck_chart(r, use_container_width=True)
                except Exception as e:
                    st.error(f"Error showing map: {e}")
                    st.write(sample_df.dtypes)
                    st.write(sample_df.head())
   
            with col2:
                try:
                    if "temp_adjusted" in sample_df.columns:
                        fig_density = px.density_mapbox(
                            sample_df, 
                            lat="latitude", 
                            lon="longitude",
                            z="temp_adjusted",
                            radius=15,
                            zoom=1,
                            mapbox_style="open-street-map",
                            title="Ocean Temperature Density Map"
                        )
                        fig_density.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_density, use_container_width=True)
                except Exception as e:
                    st.info("Density map requires valid temperature data")

    with tab2:
        st.subheader("📊 Statistical Data Distributions")
        
        col1, col2 = st.columns(2)
        with col1:
            if "temp_adjusted" in df.columns:
                try:
                    fig_temp = px.histogram(
                        df.dropna(subset=['temp_adjusted']), 
                        x="temp_adjusted", 
                        title="Ocean Temperature Distribution",
                        nbins=50,
                        color_discrete_sequence=["#667eea"]
                    )
                    fig_temp.update_layout(
                        xaxis_title="Temperature (°C)", 
                        yaxis_title="Frequency",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating temperature histogram: {e}")
        
        with col2:
            if "psal_adjusted" in df.columns:
                try:
                    fig_sal = px.histogram(
                        df.dropna(subset=['psal_adjusted']), 
                        x="psal_adjusted", 
                        title="Ocean Salinity Distribution",
                        nbins=50,
                        color_discrete_sequence=["#f5576c"]
                    )
                    fig_sal.update_layout(
                        xaxis_title="Salinity (PSU)", 
                        yaxis_title="Frequency",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_sal, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating salinity histogram: {e}")

    with tab3:
        st.subheader("🌐 3D Ocean Data Visualization")
        
        num_points = min(1000, len(df))
        float_sample = df.sample(num_points)

        points_data = []
        labels_data = []

        for _, row in float_sample.iterrows():
            points_data.append({
                'lat': float(row['latitude']),
                'lng': float(row['longitude']),
                'size': 0.01,
                'color': 'rgba(50,205,50,0.9)'
            })

        streamlit_globe(
            pointsData=points_data,
            labelsData=[],
            daytime='day',
            width=900,
            height=700
        )

    with tab4:
        st.subheader("📈 Advanced Statistical Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if all(c in df.columns for c in ["pres_adjusted", "temp_adjusted"]):
                try:
                    df_clean = df.dropna(subset=['pres_adjusted', 'temp_adjusted'])
                    df_clean = df_clean[(df_clean['pres_adjusted'] > 0) & (df_clean['temp_adjusted'] > -5)]
                    
                    if len(df_clean) > 0:
                        df_clean["depth_range"] = pd.cut(
                            df_clean["pres_adjusted"], 
                            bins=5, 
                            labels=["Surface", "Shallow", "Medium", "Deep", "Abyssal"]
                        )
                        
                        fig_box = px.box(
                            df_clean, 
                            x="depth_range", 
                            y="temp_adjusted",
                            title="Temperature Distribution by Ocean Depth",
                            color="depth_range"
                        )
                        fig_box.update_layout(
                            xaxis_title="Depth Range", 
                            yaxis_title="Temperature (°C)",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.warning("No valid data for depth analysis")
                except Exception as e:
                    st.error(f"Error creating box plot: {e}")
        
        with col2:
            if all(c in df.columns for c in ["psal_adjusted", "pres_adjusted"]):
                try:
                    df_clean = df.dropna(subset=['psal_adjusted', 'pres_adjusted'])
                    df_clean = df_clean[(df_clean['pres_adjusted'] > 0) & (df_clean['psal_adjusted'] > 0)]
                    
                    if len(df_clean) > 0:
                        df_clean["pressure_bin"] = pd.cut(
                            df_clean["pres_adjusted"], 
                            bins=4, 
                            precision=0,
                            labels=["Surface", "Mid-depth", "Deep", "Abyssal"]
                        )
                        
                        fig_violin = px.violin(
                            df_clean, 
                            x="pressure_bin", 
                            y="psal_adjusted",
                            title="Salinity Distribution by Ocean Depth",
                            box=True,
                            color="pressure_bin"
                        )
                        fig_violin.update_layout(
                            xaxis_title="Depth Range", 
                            yaxis_title="Salinity (PSU)",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis={'tickangle': 45}
                        )
                        st.plotly_chart(fig_violin, use_container_width=True)
                    else:
                        st.warning("No valid data for salinity depth analysis")
                except Exception as e:
                    st.error(f"Error creating violin plot: {e}")

    with tab5:
        st.subheader("🔬 Deep Ocean Data Analysis")
        
        # Enhanced correlation analysis
        if all(c in df.columns for c in ["psal_adjusted", "temp_adjusted", "pres_adjusted"]):
            try:
                sample_df = df.dropna(subset=["psal_adjusted", "temp_adjusted", "pres_adjusted"]).sample(min(3000, len(df)))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 3D Scatter plot
                    fig_3d = go.Figure(data=[go.Scatter3d(
                        x=sample_df["psal_adjusted"], 
                        y=sample_df["temp_adjusted"], 
                        z=-sample_df["pres_adjusted"],  # Negative for depth visualization
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=sample_df["temp_adjusted"],
                            colorscale="viridis",
                            opacity=0.8,
                            colorbar=dict(title="Temperature (°C)", x=1.1)
                        ),
                        text=[f"Temp: {t:.2f}°C<br>Sal: {s:.2f} PSU<br>Depth: {p:.1f}m" 
                              for t, s, p in zip(sample_df["temp_adjusted"], sample_df["psal_adjusted"], sample_df["pres_adjusted"])],
                        hovertemplate="%{text}<extra></extra>"
                    )])
                    
                    fig_3d.update_layout(
                        title="3D Ocean Properties (Salinity vs Temperature vs Depth)",
                        scene=dict(
                            xaxis_title="Salinity (PSU)",
                            yaxis_title="Temperature (°C)",
                            zaxis_title="Depth (negative meters)",
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        height=600,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                with col2:
                    # Temperature-Salinity diagram
                    fig_ts = px.scatter(
                        sample_df,
                        x="psal_adjusted", 
                        y="temp_adjusted",
                        color="pres_adjusted",
                        title="T-S Diagram (Temperature-Salinity Relationship)",
                        color_continuous_scale="plasma",
                        labels={"pres_adjusted": "Pressure (dbar)"}
                    )
                    fig_ts.update_layout(
                        xaxis_title="Salinity (PSU)", 
                        yaxis_title="Temperature (°C)",
                        height=600,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in deep analysis visualization: {e}")

        # Temporal analysis if available
        if 'juld' in df.columns:
            st.subheader("📅 Temporal Ocean Patterns")
            try:
                df_time = df.copy()
                df_time['date'] = pd.to_datetime(df_time['juld'], unit='s', origin='2000-01-01', errors='coerce')
                df_time = df_time.dropna(subset=['date', 'temp_adjusted'])
                
                if len(df_time) > 100:
                    # Monthly aggregation
                    df_monthly = df_time.groupby(df_time['date'].dt.to_period('M')).agg({
                        'temp_adjusted': ['mean', 'std'],
                        'psal_adjusted': 'mean'
                    }).reset_index()
                    
                    df_monthly.columns = ['date', 'temp_mean', 'temp_std', 'sal_mean']
                    df_monthly['date'] = df_monthly['date'].dt.to_timestamp()
                    
                    # Temperature trend with error bars
                    fig_time = go.Figure()
                    
                    fig_time.add_trace(go.Scatter(
                        x=df_monthly['date'],
                        y=df_monthly['temp_mean'],
                        mode='lines+markers',
                        name='Average Temperature',
                        line=dict(color='#667eea', width=3),
                        error_y=dict(
                            type='data',
                            array=df_monthly['temp_std'],
                            visible=True,
                            color='rgba(102,126,234,0.3)'
                        )
                    ))
                    
                    fig_time.update_layout(
                        title="Ocean Temperature Trends Over Time",
                        xaxis_title="Date",
                        yaxis_title="Temperature (°C)",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.info("Insufficient temporal data for trend analysis")
                    
            except Exception as e:
                st.info("Temporal analysis not available")
elif st.session_state.current_view == 'raw':
    st.markdown("## 📊 Ocean Data Overview & Analysis")
    
    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df):,}</h3>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_temp = df['temp_adjusted'].mean() if 'temp_adjusted' in df.columns and not df['temp_adjusted'].isna().all() else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_temp:.2f}°C</h3>
            <p>Avg Temperature</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_sal = df['psal_adjusted'].mean() if 'psal_adjusted' in df.columns and not df['psal_adjusted'].isna().all() else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_sal:.2f}</h3>
            <p>Avg Salinity</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        max_depth = df['pres_adjusted'].max() if 'pres_adjusted' in df.columns and not df['pres_adjusted'].isna().all() else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{max_depth:.0f} dbar</h3>
            <p>Max Depth</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Interactive Data Filtering")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'temp_adjusted' in df.columns and not df['temp_adjusted'].isna().all():
            temp_min = float(df['temp_adjusted'].min())
            temp_max = float(df['temp_adjusted'].max())
            temp_range = st.slider("Temperature Range (°C)", temp_min, temp_max, (temp_min, temp_max), key="temp_filter")
        else:
            temp_range = (0, 0)
            st.warning("Temperature data not available")
    
    with col2:
        if 'pres_adjusted' in df.columns and not df['pres_adjusted'].isna().all():
            pres_min = float(df['pres_adjusted'].min())
            pres_max = float(df['pres_adjusted'].max())
            pressure_range = st.slider("Pressure Range (dbar)", pres_min, pres_max, (pres_min, pres_max), key="pres_filter")
        else:
            pressure_range = (0, 0)
            st.warning("Pressure data not available")

    # Apply filters and display data
    try:
        filtered_df = df.copy()
        if 'temp_adjusted' in df.columns and temp_range != (0, 0):
            filtered_df = filtered_df[
                filtered_df['temp_adjusted'].between(temp_range[0], temp_range[1])
            ]
        if 'pres_adjusted' in df.columns and pressure_range != (0, 0):
            filtered_df = filtered_df[
                filtered_df['pres_adjusted'].between(pressure_range[0], pressure_range[1])
            ]
        
        st.success(f"📈 Displaying {len(filtered_df):,} of {len(df):,} records after filtering")
        st.dataframe(filtered_df, use_container_width=True, height=400)
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        st.dataframe(df.head(1000), use_container_width=True, height=400)
elif st.session_state.current_view == 'download':
    st.markdown("## 💾 Advanced Data Export System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Export Configuration")
        
        # Enhanced export options
        export_format = st.selectbox(
            "Choose Export Format", 
            ["CSV", "JSON", "Parquet", "Excel-Compatible CSV"],
            help="Select the optimal format for your analysis needs"
        )
        
        # Smart record limiting
        total_records = len(df)
        default_limit = min(50000, total_records)
        max_records = st.number_input(
            "Maximum Records to Export", 
            min_value=100, 
            max_value=total_records, 
            value=default_limit,
            step=5000,
            help=f"Limit export size (Total available: {total_records:,})"
        )
        
        # Intelligent column selection
        available_columns = df.columns.tolist()
        core_columns = [col for col in ['latitude', 'longitude', 'temp_adjusted', 'psal_adjusted', 'pres_adjusted', 'juld'] if col in available_columns]
        selected_columns = st.multiselect(
            "Select Data Columns",
            available_columns,
            default=core_columns,
            help="Choose which oceanographic parameters to include"
        )
        
        # Advanced filtering
        st.subheader("🎯 Advanced Data Filtering")
        
        filter_by_region = st.checkbox("🌍 Geographic Region Filter")
        if filter_by_region and all(c in df.columns for c in ['latitude', 'longitude']):
            col_lat1, col_lat2 = st.columns(2)
            with col_lat1:
                lat_min = st.number_input("Minimum Latitude", value=-90.0, min_value=-90.0, max_value=90.0)
                lon_min = st.number_input("Minimum Longitude", value=-180.0, min_value=-180.0, max_value=180.0)
            with col_lat2:
                lat_max = st.number_input("Maximum Latitude", value=90.0, min_value=-90.0, max_value=90.0)
                lon_max = st.number_input("Maximum Longitude", value=180.0, min_value=-180.0, max_value=180.0)
        
        # Quality control filter
        filter_by_qc = st.checkbox("🔬 Quality Control Filter")
        if filter_by_qc:
            qc_columns = [col for col in df.columns if 'qc' in col.lower()]
            if qc_columns:
                selected_qc = st.selectbox("Quality Control Column", qc_columns)
                qc_values = st.multiselect("Accepted QC Values", ['1', '2', '3', '4', '5', '8'], default=['1'])
    
    with col2:
        st.subheader("📊 Export Summary")
        
        # Apply all filters progressively
        export_df = df.copy()
        
        # Geographic filter
        if filter_by_region and all(c in df.columns for c in ['latitude', 'longitude']):
            export_df = export_df[
                (export_df['latitude'].between(lat_min, lat_max)) & 
                (export_df['longitude'].between(lon_min, lon_max))
            ]
        
        # Quality control filter
        if filter_by_qc and 'selected_qc' in locals() and 'qc_values' in locals():
            if selected_qc in export_df.columns:
                export_df = export_df[export_df[selected_qc].astype(str).isin(qc_values)]
        
        # Column selection
        if selected_columns:
            available_selected = [col for col in selected_columns if col in export_df.columns]
            if available_selected:
                export_df = export_df[available_selected]
        
        # Record limit
        export_df = export_df.head(max_records)
        
        # Summary metrics
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(export_df):,}</h3>
            <p>Records to Export</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(export_df.columns)}</h3>
            <p>Data Columns</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced size estimation
        est_size_mb = len(export_df) * len(export_df.columns) * 8 / (1024 * 1024)  # Better estimation
        st.markdown(f"""
        <div class="metric-card">
            <h3>{est_size_mb:.1f} MB</h3>
            <p>Estimated Size</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced download section
    st.markdown("### 🚀 Download Your Ocean Data")
    
    if len(export_df) > 0:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if export_format in ["CSV", "Excel-Compatible CSV"]:
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv_data,
                    f"ocean_data_{timestamp}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col2:
            json_data = export_df.to_json(orient="records", indent=2)
            st.download_button(
                "📋 Download JSON",
                json_data,
                f"ocean_data_{timestamp}.json",
                "application/json",
                use_container_width=True
            )
        
        with col3:
            # Parquet format (more efficient)
            try:
                parquet_data = export_df.to_parquet()
                st.download_button(
                    "🗜️ Download Parquet",
                    parquet_data,
                    f"ocean_data_{timestamp}.parquet",
                    "application/octet-stream",
                    use_container_width=True
                )
            except:
                st.button("🗜️ Parquet N/A", disabled=True, use_container_width=True)
        
        with col4:
            # Summary statistics
            summary_data = export_df.describe().to_csv()
            st.download_button(
                "📈 Download Summary",
                summary_data,
                f"ocean_summary_{timestamp}.csv",
                "text/csv",
                use_container_width=True
            )
    
    # Data preview
    st.subheader("👁️ Export Data Preview")
    if len(export_df) > 0:
        st.dataframe(export_df.head(15), use_container_width=True)
        st.info(f"Preview showing first 15 rows of {len(export_df):,} total export records")
    else:
        st.warning("⚠️ No data matches your current filter criteria")
elif st.session_state.current_view == 'filter':
    st.markdown("## 🌍 Regional Ocean Data Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🗺️ Ocean Region Selection")
        
        # Enhanced predefined regions with more accuracy
        regions = {
            "Global Ocean": {"lat": [-90, 90], "lon": [-180, 180], "desc": "Complete global ocean coverage"},
            "Indian Ocean": {"lat": [-60, 30], "lon": [20, 120], "desc": "Indian Ocean basin including Arabian Sea & Bay of Bengal"},
            "Pacific Ocean": {"lat": [-60, 65], "lon": [120, -70], "desc": "Pacific Ocean basin (largest ocean)"},
            "Atlantic Ocean": {"lat": [-60, 70], "lon": [-80, 20], "desc": "Atlantic Ocean basin"},
            "Southern Ocean": {"lat": [-90, -60], "lon": [-180, 180], "desc": "Antarctic/Southern Ocean"},
            "Arabian Sea": {"lat": [8, 25], "lon": [50, 78], "desc": "Northwestern Indian Ocean"},
            "Bay of Bengal": {"lat": [5, 25], "lon": [78, 100], "desc": "Northeastern Indian Ocean"},
            "Mediterranean Sea": {"lat": [30, 46], "lon": [-6, 37], "desc": "European mediterranean basin"},
            "North Atlantic": {"lat": [30, 70], "lon": [-80, 0], "desc": "North Atlantic Ocean region"},
            "Equatorial Pacific": {"lat": [-10, 10], "lon": [140, -80], "desc": "Tropical Pacific region"}
        }
        
        selected_region = st.selectbox(
            "Choose Ocean Region",
            list(regions.keys()),
            help="Select a predefined oceanographic region for analysis"
        )
        
        region_info = regions[selected_region]
        st.info(f"🌊 {region_info['desc']}")
        
        # Custom coordinates
        use_custom = st.checkbox("🎯 Custom Coordinates")
        
        if use_custom:
            st.markdown("**Custom Region Definition:**")
            lat_min = st.number_input("Minimum Latitude", value=float(region_info['lat'][0]), min_value=-90.0, max_value=90.0, step=0.1)
            lat_max = st.number_input("Maximum Latitude", value=float(region_info['lat'][1]), min_value=-90.0, max_value=90.0, step=0.1)
            lon_min = st.number_input("Minimum Longitude", value=float(region_info['lon'][0]), min_value=-180.0, max_value=180.0, step=0.1)
            lon_max = st.number_input("Maximum Longitude", value=float(region_info['lon'][1]), min_value=-180.0, max_value=180.0, step=0.1)
        else:
            lat_min, lat_max = region_info['lat']
            lon_min, lon_max = region_info['lon']
        
        # Enhanced additional filters
        st.markdown("### 🔧 Advanced Filters")
        
        # Temporal filter
        filter_by_time = st.checkbox("📅 Time Period Filter")
        if filter_by_time and 'juld' in df.columns:
            time_period = st.selectbox(
                "Select Time Period",
                ["Last 1 Year", "Last 2 Years", "Last 5 Years", "All Time", "Custom Range"]
            )
        
        # Depth stratification
        filter_by_depth = st.checkbox("🌊 Depth Layer Filter")
        if filter_by_depth and 'pres_adjusted' in df.columns:
            depth_layers = {
                "Surface (0-200m)": (0, 200),
                "Thermocline (200-1000m)": (200, 1000),
                "Intermediate (1000-2000m)": (1000, 2000),
                "Deep (2000m+)": (2000, 10000)
            }
            selected_depth = st.selectbox("Depth Layer", list(depth_layers.keys()))
            depth_min, depth_max = depth_layers[selected_depth]
        
    with col2:
        # Enhanced regional analysis
        if all(c in df.columns for c in ("latitude", "longitude")):
            try:
                # Apply geographic filter
                filtered_df = df[
                    (df['latitude'].between(lat_min, lat_max)) & 
                    (df['longitude'].between(lon_min, lon_max))
                ]
                
                # Apply additional filters
                if filter_by_depth and 'pres_adjusted' in df.columns:
                    filtered_df = filtered_df[
                        filtered_df['pres_adjusted'].between(depth_min, depth_max)
                    ]
                
                # Regional statistics with enhanced metrics
                st.subheader(f"📊 {selected_region} - Oceanographic Analysis")
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(filtered_df):,}</h3>
                        <p>Data Points</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_stat2:
                    if 'temp_adjusted' in filtered_df.columns and len(filtered_df) > 0:
                        avg_temp = filtered_df['temp_adjusted'].mean()
                        temp_std = filtered_df['temp_adjusted'].std()
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{avg_temp:.2f}°C</h3>
                            <p>Avg Temp (±{temp_std:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-card"><h3>N/A</h3><p>Temperature</p></div>', unsafe_allow_html=True)
                
                with col_stat3:
                    if 'psal_adjusted' in filtered_df.columns and len(filtered_df) > 0:
                        avg_sal = filtered_df['psal_adjusted'].mean()
                        sal_std = filtered_df['psal_adjusted'].std()
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{avg_sal:.2f}</h3>
                            <p>Avg Salinity (±{sal_std:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-card"><h3>N/A</h3><p>Salinity</p></div>', unsafe_allow_html=True)
                
                with col_stat4:
                    if 'pres_adjusted' in filtered_df.columns and len(filtered_df) > 0:
                        avg_depth = filtered_df['pres_adjusted'].mean()
                        max_depth = filtered_df['pres_adjusted'].max()
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{avg_depth:.0f}</h3>
                            <p>Avg Depth (Max: {max_depth:.0f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-card"><h3>N/A</h3><p>Depth</p></div>', unsafe_allow_html=True)
                
                # Enhanced regional visualization
                if len(filtered_df) > 0:
                    st.subheader(f"🗺️ {selected_region} - Spatial Distribution")
                    
                    # Intelligent sampling for performance
                    sample_size = min(3000, len(filtered_df))
                    plot_df = filtered_df.sample(sample_size) if len(filtered_df) > sample_size else filtered_df
                    
                    try:
                        fig_region = px.scatter_geo(
                            plot_df,
                            lat="latitude", 
                            lon="longitude", 
                            color="temp_adjusted" if "temp_adjusted" in plot_df.columns else None,
                            size="pres_adjusted" if "pres_adjusted" in plot_df.columns else None,
                            title=f"Ocean Data Distribution in {selected_region}",
                            color_continuous_scale="plasma",
                            size_max=15,
                            hover_data={
                                col: ':.2f' for col in ["temp_adjusted", "psal_adjusted", "pres_adjusted"] 
                                if col in plot_df.columns
                            }
                        )
                        
                        # Enhanced map styling
                        fig_region.update_geos(
                            projection_type="natural earth",
                            showland=True,
                            landcolor="rgb(243, 243, 243)",
                            coastlinecolor="rgb(204, 204, 204)",
                            showocean=True,
                            oceancolor="rgb(230, 245, 255)",
                            showlakes=True,
                            lakecolor="rgb(230, 245, 255)",
                            center=dict(lat=(lat_min+lat_max)/2, lon=(lon_min+lon_max)/2),
                            lonaxis_range=[max(lon_min-10, -180), min(lon_max+10, 180)],
                            lataxis_range=[max(lat_min-5, -90), min(lat_max+5, 90)]
                        )
                        fig_region.update_layout(
                            height=500,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_region, use_container_width=True)
                        
                        if sample_size < len(filtered_df):
                            st.info(f"Displaying {sample_size:,} sampled points of {len(filtered_df):,} total points for performance")
                    
                    except Exception as e:
                        st.error(f"Error creating regional map: {e}")
                    
                    # Regional data distributions
                    if len(filtered_df) > 10:
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            if "temp_adjusted" in filtered_df.columns:
                                try:
                                    fig_temp_dist = px.histogram(
                                        filtered_df.dropna(subset=['temp_adjusted']),
                                        x="temp_adjusted",
                                        title=f"Temperature Distribution in {selected_region}",
                                        color_discrete_sequence=["#667eea"],
                                        nbins=min(30, len(filtered_df)//10)
                                    )
                                    fig_temp_dist.update_layout(
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)'
                                    )
                                    st.plotly_chart(fig_temp_dist, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating temperature distribution: {e}")
                        
                        with col_chart2:
                            if "pres_adjusted" in filtered_df.columns:
                                try:
                                    fig_depth_dist = px.histogram(
                                        filtered_df.dropna(subset=['pres_adjusted']),
                                        x="pres_adjusted",
                                        title=f"Depth Distribution in {selected_region}",
                                        color_discrete_sequence=["#f5576c"],
                                        nbins=min(30, len(filtered_df)//10)
                                    )
                                    fig_depth_dist.update_layout(
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)'
                                    )
                                    st.plotly_chart(fig_depth_dist, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating depth distribution: {e}")
                
                else:
                    st.warning(f"⚠️ No oceanographic data found in {selected_region} with current filters")
            
            except Exception as e:
                st.error(f"Error in regional analysis: {e}")
                
        else:
            st.warning("Geographic coordinate columns not available in the dataset")
    
    # Filtered data table with enhanced display
    if 'filtered_df' in locals() and len(filtered_df) > 0:
        st.subheader("📋 Regional Data Sample")
        
        # Show data quality summary
        col1, col2, col3 = st.columns(3)
        with col1:
            complete_records = filtered_df.dropna().shape[0]
            st.metric("Complete Records", f"{complete_records:,}")
        with col2:
            if 'temp_adjusted' in filtered_df.columns:
                temp_coverage = (filtered_df['temp_adjusted'].notna().sum() / len(filtered_df)) * 100
                st.metric("Temperature Coverage", f"{temp_coverage:.1f}%")
        with col3:
            if 'psal_adjusted' in filtered_df.columns:
                sal_coverage = (filtered_df['psal_adjusted'].notna().sum() / len(filtered_df)) * 100
                st.metric("Salinity Coverage", f"{sal_coverage:.1f}%")
        
        # Display sample data
        display_df = filtered_df.head(100)
        st.dataframe(display_df, use_container_width=True, height=350)
        st.info(f"Showing first 100 rows of {len(filtered_df):,} filtered records from {selected_region}")

        # Temporal analysis if available
        if 'juld' in df.columns:
            st.subheader("📅 Temporal Ocean Patterns")
            try:
                df_time = df.copy()
                df_time['date'] = pd.to_datetime(df_time['juld'], unit='s', origin='2000-01-01', errors='coerce')
                df_time = df_time.dropna(subset=['date', 'temp_adjusted'])
                
                if len(df_time) > 100:
                    # Monthly aggregation
                    df_monthly = df_time.groupby(df_time['date'].dt.to_period('M')).agg({
                        'temp_adjusted': ['mean', 'std'],
                        'psal_adjusted': 'mean'
                    }).reset_index()
                    
                    df_monthly.columns = ['date', 'temp_mean', 'temp_std', 'sal_mean']
                    df_monthly['date'] = df_monthly['date'].dt.to_timestamp()
                    
                    # Temperature trend with error bars
                    fig_time = go.Figure()
                    
                    fig_time.add_trace(go.Scatter(
                        x=df_monthly['date'],
                        y=df_monthly['temp_mean'],
                        mode='lines+markers',
                        name='Average Temperature',
                        line=dict(color='#667eea', width=3),
                        error_y=dict(
                            type='data',
                            array=df_monthly['temp_std'],
                            visible=True,
                            color='rgba(102,126,234,0.3)'
                        )
                    ))
                    
                    fig_time.update_layout(
                        title="Ocean Temperature Trends Over Time",
                        xaxis_title="Date",
                        yaxis_title="Temperature (°C)",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.info("Insufficient temporal data for trend analysis")
                    
            except Exception as e:
                st.info("Temporal analysis not available")