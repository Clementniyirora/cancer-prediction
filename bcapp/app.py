import pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    print("Getting clean data...")
    df = pd.read_csv(r'C:\Users\user\Downloads\breastcancer.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df.diagnosis = [1 if value == "M" else 0 for value in df.diagnosis]
    return df



def add_sidebar():
    st.sidebar.header("Cell Details")

    df = get_clean_data()

    slider_labels = slider_labels = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave Points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal Dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"),
    ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"),
    ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"),
    ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"),
    ("Concave Points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"),
    ("Fractal Dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave Points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal Dimension (worst)", "fractal_dimension_worst")
]
    
    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(df[key].max()),
            value=float(df[key].mean())
        )

    return input_dict




def get_scaled_values(input_dict):
    df = get_clean_data()

    X = df.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict




def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)


    categories = ['Radius','Texture','Perimeter','Area',
                  'Smoothness','Compactness','Concavity','Concave Points',
                  'Symetery','Fractal Dimension']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
      r=[
        input_data['radius_mean'], input_data['texture_mean'],input_data["radius_mean"], 
        input_data["texture_mean"],input_data["perimeter_mean"],input_data["area_mean"],
        input_data["smoothness_mean"],input_data["compactness_mean"],input_data["concavity_mean"],
        input_data["concave points_mean"],input_data["symmetry_mean"],input_data["fractal_dimension_mean"]
      ],
      theta=categories,
      fill='toself',
      name= 'Mean Value'
      ))
    fig.add_trace(go.Scatterpolar(
      r=[
        input_data['radius_se'], input_data['texture_se'],input_data["radius_se"], 
        input_data["texture_se"],input_data["perimeter_se"],input_data["area_se"],
        input_data["smoothness_se"],input_data["compactness_se"],input_data["concavity_se"],
        input_data["concave points_se"],input_data["symmetry_se"],input_data["fractal_dimension_se"]  
      ],
      theta=categories,
      fill='toself',
      name='Standard Error'
      ))
    
    fig.add_trace(go.Scatterpolar(
      r=[
        input_data['radius_worst'], input_data['texture_worst'],input_data["radius_worst"], 
        input_data["texture_worst"],input_data["perimeter_worst"],input_data["area_worst"],
        input_data["smoothness_worst"],input_data["compactness_worst"],input_data["concavity_worst"],
        input_data["concave points_worst"],input_data["symmetry_worst"],input_data["fractal_dimension_worst"]  
      ],
      theta=categories,
      fill='toself',
      name='Worst Value'
      ))


    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
                )),
            showlegend=True
    )

    return fig




def add_predictions(input_data):
    lr = pickle.load(open("model/lr.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1,-1)

    input_df = pd.DataFrame(input_array, columns=input_data.keys())

    input_array_scaled = scaler.transform(input_df)

    prediction = lr.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.markdown("<p style='color: red; font-weight: bold;'>BENIGN</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: blue; font-weight: bold;'>MALICIOUS</p>", unsafe_allow_html=True)



    st.write("Probability of being benign: ", lr.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", lr.predict_proba(input_array_scaled)[0][1])
    st.write("**Disclaimer:** This app is intended to support healthcare decision-making but should not be used as a substitute for professional medical diagnosis.")



def ml():
    st.set_page_config(
        page_title= "Breast Cancer Predictor",
        page_icon= ":female-doctor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar()

    with st.container():
        st.title("CNBIO Breast Cancer Predictor")
        st.write("Improve your healthcare decisions with CNBIO app designed to predict breast cancer diagnosis. Utilizing machine learning model, this app analyzes cytosis measurements to distinguish between benign and malignant cases.")
    
    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart= get_radar_chart (input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)









if __name__ == '__main__':
    ml()

