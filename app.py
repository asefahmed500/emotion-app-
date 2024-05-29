import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import pytz
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST
# Load Model
pipe_lr = joblib.load(open("model.pkl", "rb"))

# Define the array of emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'love', 'sadness', 'shame', 'surprise']

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱",
    "guilt": "😔", "joy": "😂", "love": "❤️",
    "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Main Application
def main():
    st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .sidebar .sidebar-content {
            padding: 10px;
        }
        .sidebar .sidebar-content h3 {
            margin-top: 0;
        }
        .main .block-container {
            padding-top: 20px;
        }
        .stTextArea textarea {
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Emotion Detector ")
    
    with st.sidebar:
        st.header("Menu")
        menu = ["Home", "Monitor"]
        choice = st.selectbox("", menu)

    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here", placeholder="Enter text to analyze emotion...")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction_index = predict_emotions(raw_text)
            prediction = emotion_labels[prediction_index]
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                predicted_emotion = emotions_emoji_dict.get(prediction, "Unknown")
                emoji_icon = emotions_emoji_dict.get(prediction, "❓")
                st.write(f"Emotion: {predicted_emotion} {emoji_icon}")
                st.write(f"Confidence: {np.max(probability):.2f}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x=alt.X('emotions', sort=None),
                    y='probability',
                    color='emotions'
                ).properties(width=350, height=300)
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(
                x=alt.X('Page Name', sort=None),
                y='Counts',
                color='Page Name'
            ).properties(width=350, height=300)
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(
                x=alt.X('Prediction', sort=None),
                y='Counts',
                color='Prediction'
            ).properties(width=350, height=300)
            st.altair_chart(pc, use_container_width=True)

    st.markdown("Designed by Asef Ahmed", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
