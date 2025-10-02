import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import shap
import nltk # Make sure nltk is imported

# Import our custom modules
import model_engine
import nlp_insight

# --- Page Configuration ---
st.set_page_config(page_title="Project Nirmaan", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")

# --- Caching Models for Performance ---
@st.cache_resource
def load_models():
    return model_engine.get_trained_models()

# --- Main App Logic ---
def main():
    # --- ROBUST NLTK SETUP ---
    # This code runs only once per session and is very reliable.
    if 'nltk_downloaded' not in st.session_state:
        with st.spinner("Downloading necessary language models... (This is a one-time setup)"):
            try:
                nltk.download('vader_lexicon')
                st.session_state['nltk_downloaded'] = True
            except Exception as e:
                st.error(f"Error downloading NLTK data: {e}")
                st.stop()

    cost_model, timeline_model, feature_names = load_models()
    
    # --- Sidebar Navigation ---
    st.sidebar.title("🏗️ Project Nirmaan")
    st.sidebar.markdown("### National Infrastructure Command Center")
    page = st.sidebar.selectbox("Navigate", ["📊 Live Project Dashboard", "🔮 What-If Simulator", "📜 Vendor Report Analyzer"])

    if page == "📊 Live Project Dashboard":
        show_dashboard()
    elif page == "🔮 What-If Simulator":
        show_simulator(cost_model, timeline_model)
    elif page == "📜 Vendor Report Analyzer":
        show_nlp_analyzer()

# --- Page 1: The Dashboard ---
def show_dashboard():
    st.header("📊 Live Project Risk Dashboard")
    st.markdown("Real-time overview of all POWERGRID projects across India.")
    
    projects = [
        {'name': 'Mumbai Substation', 'lat': 19.07, 'lon': 72.87, 'risk': 'High'},
        {'name': 'Delhi Overhead Line', 'lat': 28.70, 'lon': 77.10, 'risk': 'Medium'},
        {'name': 'Bangalore UG Cable', 'lat': 12.97, 'lon': 77.59, 'risk': 'Low'},
        {'name': 'Kolkata Grid', 'lat': 22.57, 'lon': 88.36, 'risk': 'High'},
        {'name': 'Chennai Plant', 'lat': 13.08, 'lon': 80.27, 'risk': 'Medium'}
    ]
    df_map = pd.DataFrame(projects)
    
    fig_map = px.scatter_mapbox(df_map, lat="lat", lon="lon", color="risk", hover_name="name",
                                color_discrete_map={"Low":"green", "Medium":"orange", "High":"red"},
                                zoom=4, height=500, mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Projects", "247", "12 this month")
    col2.metric("At-Risk Projects", "38", "-5% from last week")
    col3.metric("Avg. Predicted Delay", "45 days", "📈")
    
    trend_data = pd.DataFrame({
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'high_risk': [20, 25, 22, 30, 35, 38],
        'medium_risk': [50, 55, 60, 58, 62, 65],
        'low_risk': [100, 95, 98, 92, 90, 88]
    })
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=trend_data['month'], y=trend_data['high_risk'], mode='lines+markers', name='High Risk', line=dict(color='red')))
    fig_trend.add_trace(go.Scatter(x=trend_data['month'], y=trend_data['medium_risk'], mode='lines+markers', name='Medium Risk', line=dict(color='orange')))
    fig_trend.add_trace(go.Scatter(x=trend_data['month'], y=trend_data['low_risk'], mode='lines+markers', name='Low Risk', line=dict(color='green')))
    st.subheader("Project Risk Trend (Last 6 Months)")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- Page 2: The Simulator ---
def show_simulator(cost_model, timeline_model):
    st.header("🔮 Project Risk Simulator")
    st.markdown("Model a new project and predict its risk profile. Identify hotspots before they become problems.")
    
    with st.form("simulator_form"):
        st.subheader("Input Project Parameters")
        col1, col2 = st.columns(2)
        with col1:
            project_type = st.selectbox("Project Type", ['Substation', 'Overhead Line', 'UG Cable'])
            terrain = st.selectbox("Terrain", ['Plain', 'Hilly', 'Forest'])
            budget = st.number_input("Budgeted Cost (in Crores)", 50, 1000, 200)
            vendor_score = st.slider("Vendor Performance Score (1-10)", 1, 10, 7)
        with col2:
            monsoon_days = st.slider("Expected Monsoon Days", 30, 120, 60)
            material_index = st.slider("Material Availability Index (1-10)", 1, 10, 8)
            regulatory_level = st.selectbox("Regulatory Hindrance Level", ['Low', 'Medium', 'High'])
            manpower_score = st.slider("Skilled Manpower Score (1-10)", 1, 10, 8)
        
        submitted = st.form_submit_button("🚀 Run Prediction")
        
        if submitted:
            input_df = pd.DataFrame({
                'project_type': [project_type], 'terrain': [terrain], 'budgeted_cost_crores': [budget],
                'vendor_performance_score': [vendor_score], 'monsoon_days': [monsoon_days],
                'material_availability_index': [material_index], 'regulatory_hindrance_level': [regulatory_level],
                'skilled_manpower_score': [manpower_score]
            })
            
            cost_pred = cost_model.predict(input_df)[0]
            timeline_pred = timeline_model.predict(input_df)[0]
            
            st.success("Prediction Complete!")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("💰 Predicted Cost Overrun", f"{cost_pred:.2f}%")
            col_res2.metric("⏱️ Predicted Timeline Delay", f"{timeline_pred:.0f} Days")
            
            st.subheader("🔍 Hotspot Analysis (Root Cause of Risk)")
            explainer = shap.Explainer(cost_model.named_steps['model'], cost_model.named_steps['preprocessor'].transform(input_df))
            shap_values = explainer(cost_model.named_steps['preprocessor'].transform(input_df))
            
            fig_shap = plt.figure()
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            st.pyplot(fig_shap, use_container_width=True)
            
            st.subheader("💡 Prescriptive Actions")
            if vendor_score < 5:
                st.error("🔴 **Action:** Vendor performance is low. Consider re-evaluating the contract or increasing oversight.")
            if terrain == 'Forest':
                st.warning("🟡 **Action:** Forest terrain requires specialized clearances. Engage with environmental agencies early.")
            if material_index < 6:
                st.warning("🟡 **Action:** Material availability is a concern. Secure alternative suppliers now.")

# --- Page 3: The NLP Analyzer ---
def show_nlp_analyzer():
    st.header("📜 Vendor & Stakeholder Report Analyzer")
    st.markdown("Uncover hidden risks from unstructured text reports, emails, and meeting minutes.")
    
    vendor_report = st.text_area("Paste Vendor Report Text Here:", height=200, placeholder="e.g., 'The delivery of steel girders is facing a 2-week delay due to a shortage at the supplier's end. We are trying to resolve this issue.'")
    
    if st.button("Analyze Report"):
        if vendor_report:
            analysis = nlp_insight.analyze_vendor_report(vendor_report)
            
            st.subheader("Analysis Results")
            sentiment_score = analysis['sentiment']
            if sentiment_score < -0.2:
                st.error(f"🔴 Negative Sentiment Detected (Score: {sentiment_score:.2f})")
            elif sentiment_score > 0.2:
                st.success(f"🟢 Positive Sentiment Detected (Score: {sentiment_score:.2f})")
            else:
                st.warning(f"🟡 Neutral Sentiment Detected (Score: {sentiment_score:.2f})")
            
            if analysis['risks_found']:
                st.subheader("⚠️ Key Risk Keywords Found:")
                st.write(analysis['risks_found'])
            else:
                st.info("No immediate risk keywords were found.")
            
            st.subheader("Keyword Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(vendor_report)
            fig_wc, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc, use_container_width=True)
        else:
            st.warning("Please enter some text to analyze.")

# --- Run the App ---
if __name__ == "__main__":
    main()