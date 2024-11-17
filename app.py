import streamlit as st
import cv2
import numpy as np
import imutils
import easyocr
import pickle
from PIL import Image

st.title("License Plate Detection App")
st.write("Upload an image to detect the license plate text and save the processed data.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and process the uploaded image
    image = Image.open(uploaded_file)
    # Convert PIL Image to RGB then to BGR for OpenCV compatibility
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Display original image
    st.image(image, caption="Original Image", use_container_width=True)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter and Canny edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    
    # Find contours and locate the license plate
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    # If a license plate location is found, proceed with masking and text extraction
    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        
        (x, y) = np.where(mask == 255)
        if len(x) > 0 and len(y) > 0:  # Check if coordinates were found
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
            
            # Display cropped license plate region
            st.image(cropped_image, caption="Cropped License Plate Region", use_container_width=True)
            
            try:
                # OCR for text recognition
                reader = easyocr.Reader(['en'])
                result = reader.readtext(cropped_image)
                
                # Display the detected text
                if result:
                    text = result[0][-2]
                    st.write("Detected License Plate Text:", text)
                    
                    # Convert BGR back to RGB for displaying with streamlit
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Annotate the image
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    annotated_img = cv2.putText(img_rgb.copy(), text=text, 
                                              org=(location[0][0][0], location[1][0][1] + 60),
                                              fontFace=font, fontScale=1, color=(0, 255, 0), 
                                              thickness=2, lineType=cv2.LINE_AA)
                    annotated_img = cv2.rectangle(annotated_img, tuple(location[0][0]), 
                                                tuple(location[2][0]), (0, 255, 0), 3)
                    
                    # Display annotated image
                    st.image(annotated_img, caption="Annotated Image", use_container_width=True)
                    
                    # Save processed data to a pickle file
                    processed_data = {'location': location, 'text': text}
                    with open('processed_data.pkl', 'wb') as file:
                        pickle.dump(processed_data, file)
                    
                    st.write("Processed data saved to `processed_data.pkl`.")
                else:
                    st.write("No text detected on the license plate.")
            except Exception as e:
                st.error(f"Error during OCR processing: {str(e)}")
        else:
            st.write("Invalid mask coordinates generated.")
    else:
        st.write("Could not detect a license plate in the image.")
