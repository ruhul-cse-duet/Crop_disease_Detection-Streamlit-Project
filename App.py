import streamlit as st
import torch
import numpy as np
import pandas as pd

from torchvision import transforms, models
import cv2
from PIL import Image
from CBAM_model import prediction_image
from custom_resnet import prediction_img

from Treatment import treatment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                       'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                       'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
                       'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
                       'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
                         'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                         'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy',
                           'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
                           'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
                           'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']


#Sidebar.............................................................
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

if(app_mode == "Home"):
    st.header("CROPS DISEASE PREDICTION AND MANAGEMENT SYSTEM")
    image_path = 'home_page.jpeg'
    st.image(image_path, use_container_width=True)

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif(app_mode == 'About'):
    st.header("About Our Project")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)
    # Create structured data for table
    data = []
    for cls in class_name:
        parts = cls.split("___")
        crop = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"
        data.append({"ফসলের নাম (Crop)": crop, "রোগ/অবস্থা (Condition)": condition})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Streamlit app
    st.set_page_config(page_title="ফসল ও রোগের তালিকা", layout="wide")
    st.markdown(
        "<h3 style='text-align: center; color: yellow;'>🌿 List of Every Crops with Disease</h3>",
        unsafe_allow_html=True
    )

    st.dataframe(df, use_container_width=True)


elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        test_image = Image.open(test_image).convert('RGB')
        #test_image = cv2.resize(test_image, (512, 512))
        st.image(test_image, caption="Uploaded Image", width=400)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # for moving data to device (CPU or GPU)
        def to_device(data, device):
            """Move tensor(s) to chosen device"""
            if isinstance(data, (list,tuple)):
                return [to_device(x, device) for x in data]
            return data.to(device, non_blocking=True)
        

        image = transform(test_image)
        image = image.unsqueeze(0)  # Add batch dimension [1, 3, 224, 224]
        image = to_device(image, device)
        #image = image.to(device)
    else:
        st.warning("Please upload an image file to continue.")

    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result = prediction_img(image)
        #Reading Labels


        st.success(f"Predicted Class is --->  {class_name[result]}")
        category =[]
        for i in class_name:
            category.append(i)
        for i in range(len(class_name)):
            if (i == result):
                output = category[i]
                break

        treatment(output)

        #  streamlit run App.py







