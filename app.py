import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Lung Health Assistant", layout="wide")

st.title("🫁 Lung Disease Detection & Medical Assistance")

classes = ["Normal", "Pneumonia", "Other Lung Disease"]

file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=350)

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    # Example prediction (replace with your real model)
    prediction = np.random.rand(3)

    index = prediction.argmax()
    disease = classes[index]

    st.subheader("🧠 AI Diagnosis")

    if disease == "Normal":

        st.success("No lung disease detected")

    else:

        st.error(f"Disease detected: {disease}")

        st.write("⚕ Please consult a pulmonologist immediately.")

        st.subheader("📍 Nearby Hospitals")

        st.components.v1.html("""
        <div id="map" style="width:100%;height:500px;"></div>

        <script>

        function initMap(lat, lon) {

            var location = {lat: lat, lng: lon};

            var map = new google.maps.Map(document.getElementById("map"), {
                zoom: 14,
                center: location
            });

            var request = {
                location: location,
                radius: '5000',
                type: ['hospital']
            };

            var service = new google.maps.places.PlacesService(map);

            service.nearbySearch(request, function(results, status) {

                if (status === google.maps.places.PlacesServiceStatus.OK) {

                    for (var i = 0; i < results.length; i++) {

                        new google.maps.Marker({
                            position: results[i].geometry.location,
                            map: map,
                            title: results[i].name
                        });

                    }

                }

            });

        }

        navigator.geolocation.getCurrentPosition(function(position) {

            var lat = position.coords.latitude;
            var lon = position.coords.longitude;

            var script = document.createElement('script');
            script.src = "https://maps.googleapis.com/maps/api/js?key=AIzaSyCbxuqZwVoUx7ItP-HsPY-bXvk8V3Q7ZGE&libraries=places&callback=initMap.bind(null,"+lat+","+lon+")";
            script.async = true;

            document.head.appendChild(script);

        });

        </script>
        """, height=520)
