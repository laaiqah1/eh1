import streamlit as st
import boto3
import sagemaker
from PIL import Image
import io
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import DataSerializer

# Function to load the SageMaker model
def load_model():
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

    # Hugging Face model configuration
    hub = {
        'HF_MODEL_ID': 'shadowlilac/aesthetic-shadow-v2',
        'HF_TASK': 'image-classification'
    }

    # Create Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        transformers_version='4.37.0',
        pytorch_version='2.1.0',
        py_version='py310',
        env=hub,
        role=role,
    )

    # Deploy the model
    predictor = huggingface_model.deploy(
        initial_instance_count=1,  # Number of instances
        instance_type='ml.m5.xlarge'  # EC2 instance type
    )

    # Set up the serializer
    predictor.serializer = DataSerializer(content_type='image/x-image')

    return predictor

# Streamlit UI
st.title('Hugging Face Image Classification')
st.write("Upload an image to classify it using the Hugging Face model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to byte array
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='JPEG')
    img_byte_array = img_byte_array.getvalue()

    # Load model and make predictions
    if st.button('Classify Image'):
        # Load the SageMaker model (assuming it's already deployed)
        predictor = load_model()

        # Predict with the model
        prediction = predictor.predict(img_byte_array)
        st.write(f"Prediction: {prediction}")
