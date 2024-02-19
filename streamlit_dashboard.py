import streamlit as st
import numpy as np
from PIL import Image
import requests
from utils import preprocess_image

model_endpoint = "http://localhost:7777/invocations"

# Title
st.title('HASH Project Dashboard')

# Sidebar
page = st.sidebar.selectbox('Navigation', ["Prediction", "Model analysis"])
st.sidebar.markdown("""---""")
st.sidebar.write("PARTNERS")
st.sidebar.image("marc.jpg", width=100)
st.sidebar.image("ailab.jpg", width=100)
st.sidebar.image("mak.jpg", width=100)
st.sidebar.image("hash.jpg", width=100)

# Parameter initialization
submit = None
uploaded_file = None

if page == "Prediction":
    # Inputs
    st.markdown("Select input ultrasound image.")
    upload_columns = st.columns([2, 1])
    
    try:
        # File upload
        file_upload = upload_columns[0].expander(label="Upload an image file.")
        uploaded_file = file_upload.file_uploader("Choose an image file", type=['jpg','png','jpeg'])

        # Validity Check
        if uploaded_file is None:
            st.error("No image uploaded :no_entry_sign:")
        if uploaded_file is not None:
            st.info("Image uploaded successfully :ballot_box_with_check:")

            # Open the image using Pillow
            image = Image.open(uploaded_file)
            upload_columns[1].image(image,caption="Uploaded Image")
            submit = upload_columns[1].button("Submit Image")

    except Exception as e:
        st.error(f"Error during file upload: {str(e)}") 

    # Data Submission
    st.markdown("""---""")
    if submit:
        try:
            with st.spinner(text="Fetching model prediction..."):
                # Preprocess Input Image
                array = preprocess_image(image)
                # Image Request
                image_request = {
                "instances":array.tolist()}
                # Response
                response = requests.post(model_endpoint, json=image_request)
                # Model Predictions
                probabilities = eval(response.text)["predictions"]
                prediction = np.argmax(probabilities,axis=1)
            # Ouputs
            outputs = st.columns([2, 1])
            outputs[0].markdown("LUS Pathology Prediction: ")

            if prediction == 0:
                outputs[1].success("myoma")
            elif prediction == 1:
                outputs[1].success("healthy")
            elif prediction == 2:
                outputs[1].success("unknown")
            else:
                outputs[1].error("Error: Invalid Outcome")

            prediction_details = st.expander(label="Model details")
            details = prediction_details.columns([2, 1])

            # All of this is mocked
            details[0].markdown("PATHOLOGY")
            details[0].markdown("MYOMA")
            details[0].markdown("HEALTHY")
            details[0].markdown("UNKNOWN")
            details[1].markdown("CONFIDENCE")
            details[1].markdown("{:.2f}%".format(100*probabilities[0][0]))
            details[1].markdown("{:.2f}%".format(100*probabilities[0][1]))
            details[1].markdown("{:.2f}%".format(100*probabilities[0][2]))
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

else:
    st.markdown("This page is not implemented yet :no_entry_sign:")



