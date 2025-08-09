import streamlit as st
import face_recognition
import numpy as np
import cv2

st.title("Gesichtsvergleich Web-App")

uploaded_file1 = st.file_uploader("Bild 1 hochladen", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Bild 2 hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file1 and uploaded_file2:
    img1 = face_recognition.load_image_file(uploaded_file1)
    img2 = face_recognition.load_image_file(uploaded_file2)

    encodings1 = face_recognition.face_encodings(img1)
    encodings2 = face_recognition.face_encodings(img2)

    face_locations1 = face_recognition.face_locations(img1)
    face_locations2 = face_recognition.face_locations(img2)

    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    for (top, right, bottom, left) in face_locations1:
        cv2.rectangle(img1_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    for (top, right, bottom, left) in face_locations2:
        cv2.rectangle(img2_bgr, (left, top), (right, bottom), (255, 0, 0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bild 1 - erkannte Gesichter")
        st.image(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB))
    with col2:
        st.subheader("Bild 2 - erkannte Gesichter")
        st.image(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB))

    if len(encodings1) == 0:
        st.warning("Kein Gesicht in Bild 1 gefunden.")
    if len(encodings2) == 0:
        st.warning("Kein Gesicht in Bild 2 gefunden.")

    if len(encodings1) > 0 and len(encodings2) > 0:
        st.subheader("Gesichtsvergleich Ergebnisse:")
        for i, enc1 in enumerate(encodings1):
            distances = face_recognition.face_distance(encodings2, enc1)
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < 0.6:  # Threshold kann angepasst werden
                st.write(f"Gesicht {i+1} aus Bild 1 stimmt überein mit Gesicht {best_match_index+1} aus Bild 2 (Distanz: {distances[best_match_index]:.2f}).")
            else:
                st.write(f"Gesicht {i+1} aus Bild 1 hat keine passende Übereinstimmung in Bild 2.")