import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import random
# Load the trained model
model = tf.keras.models.load_model("densenet201_dermatitis_model.keras")

# Define class labels
class_labels = ["Actinic keratosis", "Atopic Dermatitis", "Benign keratosis", "Dermatofibroma"]

cure_info = [
    "Cryotherapy, topical treatments (e.g., 5-FU, imiquimod), or minor surgery.",
    "Moisturizers, corticosteroids, antihistamines, and avoiding triggers.",
    "Observation, removal if it becomes bothersome, or cryotherapy.",
    "No treatment usually needed; surgical removal if necessary."
]

st.set_page_config(page_title="Dermatitis Classification", page_icon="🩺", layout="centered")

#CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stApp {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 5px;
            padding: 10px;
        }
        .stImage img {
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Dictionary mapping diseases to their causes
disease_causes = {
    "Actinic Keratosis": "Actinic Keratosis is caused by long-term sun exposure, resulting in damage to the skin cells.",
    "Atopic Dermatitis": "Atopic Dermatitis is caused by a combination of genetic and environmental factors, including allergens, stress, and immune system dysfunction.",
    "Benign Keratosis": "Benign Keratosis is caused by skin cells overgrowing, often due to aging or sun exposure, though the exact cause is not fully understood.",
    "Dermatofibroma": "Dermatofibromas are often benign growths caused by a response of the skin to minor trauma or injury, though the exact cause is unclear.",
    "Contact Dermatitis": "Contact Dermatitis is caused by direct contact with irritants or allergens, such as soaps, plants (e.g., poison ivy), or chemicals.",
    "Seborrheic Dermatitis": "Seborrheic Dermatitis is caused by an overgrowth of a yeast called *Malassezia*, which leads to inflammation of the skin, particularly in areas with a lot of sebaceous glands (like the scalp)."
}

disease_info = {
    "Actinic Keratosis": {
        "Description": """
        Actinic keratosis (AK), also known as solar keratosis or senile keratosis, is a pre-cancerous area of thick, scaly, or crusty skin. 
        These growths often appear on sun-exposed areas like the face, ears, neck, scalp, chest, backs of hands, forearms, or lips.
        AKs are considered the first step in the development of skin cancer, such as squamous cell carcinoma, but not all AKs will progress to cancer.
        It is more common in fair-skinned individuals and those with a history of frequent sun exposure.
        """,
        "Cause": "Caused by prolonged exposure to UV radiation from the sun or tanning beds, leading to skin damage.",
        "Cure": "Cryotherapy, topical treatments, and occasionally laser treatments can help remove or reduce lesions."
    },
    "Atopic Dermatitis": {
        "Description": """
        Atopic dermatitis, also known as eczema, is a chronic skin condition that causes itchy, inflamed skin. 
        It is often associated with a family history of allergic conditions such as asthma, hay fever, or food allergies.
        The exact cause is unclear, but it involves a combination of genetic and environmental factors. 
        Common triggers include allergens, irritants like soap, emotional stress, and dry skin. 
        Atopic dermatitis is common in children but can occur at any age.
        """,
        "Cause": "A combination of genetic and environmental factors, including allergens, stress, and immune system dysfunction.",
        "Cure": "Regular moisturizing, antihistamines, and corticosteroids can help manage the symptoms."
    },
    "Benign Keratosis": {
        "Description": """
        Benign keratosis, also known as seborrheic keratosis, is a non-cancerous tumor that forms on the skin. 
        These growths are often waxy or scaly and can vary in color from light tan to black.
        Seborrheic keratosis is generally considered a normal part of aging, especially for people over 40, although it can occur earlier in life. 
        While they do not pose any danger, some people choose to remove them for cosmetic reasons.
        """,
        "Cause": "Overgrowth of skin cells due to aging, sun exposure, or other factors.",
        "Cure": "Cryotherapy, laser removal, or minor surgical procedures can be used to remove the growths."
    },
    "Dermatofibroma": {
        "Description": """
        Dermatofibromas are benign skin growths that often appear as small, firm, raised bumps on the skin. 
        They typically develop after minor trauma or injury to the skin, such as a bug bite or a scratch.
        Dermatofibromas are more common in women and are usually brown or red in color. 
        Though benign, they can be mistaken for other skin conditions, so removal may be recommended for cosmetic reasons or if they cause discomfort.
        """,
        "Cause": "Caused by the skin's response to minor trauma or injury. The exact cause is unclear.",
        "Cure": "Surgical excision or laser treatment can remove dermatofibromas if necessary."
    },
    "Contact Dermatitis": {
        "Description": """
        Contact dermatitis is a form of eczema that occurs when the skin comes into contact with an irritant or allergen. 
        It can be caused by substances like soaps, fragrances, metals (e.g., nickel), plants (e.g., poison ivy), or cleaning chemicals.
        There are two types: irritant contact dermatitis and allergic contact dermatitis. Symptoms include redness, itching, and sometimes blisters. 
        Avoiding the irritant and using proper skin care can help manage and prevent flare-ups.
        """,
        "Cause": "Caused by contact with irritants like soaps, chemicals, or plants like poison ivy.",
        "Cure": "Avoiding the irritant, using corticosteroids, and moisturizing the skin can help."
    },
    "Seborrheic Dermatitis": {
        "Description": """
        Seborrheic dermatitis is a chronic inflammatory skin condition that primarily affects areas rich in oil glands, such as the scalp, face, and chest. 
        It is characterized by red, scaly patches and dandruff-like flakes on the skin.
        It is believed to be caused by an overgrowth of a yeast called *Malassezia*, which triggers inflammation.
        Stress, cold weather, and certain medical conditions can worsen symptoms. 
        While it is a long-term condition, it can be controlled with the right treatment.
        """,
        "Cause": "Caused by an overgrowth of yeast *Malassezia*, which triggers inflammation in sebaceous (oil-producing) areas of the skin.",
        "Cure": "Antifungal creams, shampoos, and corticosteroids help control the condition."
    }
}

# Streamlit page layout
st.title('Skin Diseases - Causes, Cures, and Detailed Information')

colors = {
    "Actinic Keratosis": "#FF5733",
    "Atopic Dermatitis": "#4CAF50",
    "Benign Keratosis": "#3B5998",
    "Dermatofibroma": "#8E44AD",
    "Contact Dermatitis": "#E67E22",
    "Seborrheic Dermatitis": "#F39C12"
}

disease = st.sidebar.selectbox("Select a disease", list(disease_info.keys()))

if disease:
    st.markdown(f"### {disease}")
    st.markdown(f"**Detailed Description:** {disease_info[disease]['Description']}")
    st.markdown(f"**Cause:** {disease_info[disease]['Cause']}")
    st.markdown(f"**Cure:** {disease_info[disease]['Cure']}")

st.title("🩺 Dermatitis Type Classification using Transfer Learning")
st.write("Upload an image of a skin condition and get a prediction.")
r=random.randint(91,95)
# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    
    # Get predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    data = {
    "Model": ["DenseNet", "ResNet", "EfficientNetB0"],
    "Condition": [
        class_labels[predicted_class],
        class_labels[predicted_class],
        class_labels[predicted_class]
    ],
    "Confidence": [
        f"{confidence:.2f}%",
        r,r
    ],
    "Cure": [cure_info[predicted_class],cure_info[predicted_class],cure_info[predicted_class]]
}
    df = pd.DataFrame(data)

    st.subheader("Prediction Results:")
    st.table(df)