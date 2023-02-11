import streamlit as st
import requests
import joblib
from PIL import Image

# Load and set images
header_images = Image.open('assets/diabetes.png')
st.image(header_images)

# Add information about service
st.title("Diabetes Prediction")
st.subheader("Masukkan nilai di masing-masing variabel di bawah ini. Lalu, klik tombol Predict")

# Create form of input

with st.form(key = "diabetes_data_form"):
    # create box for number input
    Pregnancies = st.number_input(
        label = "1.\tMasukkan nilai pregnancies:",
        min_value = 0,
        max_value = 100,
        help = "Nilai berkisar antara 0 sampai 100"
    )
    Glucose = st.number_input(
        label = "2.\tMasukkan nilai glucose:",
        min_value = 0,
        max_value = 200,
        help = "Nilai berkisar antara 0 sampai 200"
    )
    
    BloodPressure = st.number_input(
        label = "3.\tMasukkan nilai blood pressure:",
        min_value = 0,
        max_value = 130,
        help = "Nilai berkisar antara 0 sampai 130"
    )

    SkinThickness = st.number_input(
        label = "4.\tMasukkan nilai ketebalan kulit",
        min_value = 0,
        max_value = 100,
        help = "Nilai berkisar antara 0 sampai 100"
    )

    Insulin = st.number_input(
        label = "5.\tMasukkan nilai insulin:",
        min_value = 0,
        max_value = 850,
        help = "Nilai berkisar antara 0 sampai 850"
    )

    Age = st.number_input(
        label = "6.\tMasukkan umur:",
        min_value = 14,
        max_value = 90,
        help = "Umur berkisar antara 14 sampai 90"
    )

    BMI = st.number_input(
        label = "7.\tMasukkan nilai BMI:",
        min_value = 0,
        max_value = 100,
        help = "Nilai berkisar antara 0 sampai 100"
    )

    DiabetesPedigreeFunction = st.number_input(
        label = "8.\tMasukkan nilai diabetes pedigree function:",
        min_value = 0,
        max_value = 5,
        help = "Nilai berkisar antara 0 sampai 5"
    )

    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "Age": Age,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict", json = raw_data).json()
            
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Diindikasi tidak terkena penyakit diabetes.":
                st.warning("Diindikasi terkena penyakit diabetes.")
            else:
                st.success("Diindikasi tidak terkena penyakit diabetes.")

