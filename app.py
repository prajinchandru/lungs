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

    # Example AI prediction
    prediction = np.random.rand(3)
    index = prediction.argmax()
    disease = classes[index]

    st.subheader("🧠 AI Diagnosis")

    if disease == "Normal":

        st.success("No lung disease detected")

    else:

        st.error(f"Disease detected: {disease}")
        st.write("⚕ Please consult a **Pulmonologist** immediately.")

        st.subheader("📍 Nearby Lung Specialists")

        st.components.v1.html("""
        <div id="results"></div>

        <script>

        function getDistance(lat1, lon1, lat2, lon2) {
            const R = 6371;
            const dLat = (lat2-lat1) * Math.PI/180;
            const dLon = (lon2-lon1) * Math.PI/180;

            const a =
                Math.sin(dLat/2) * Math.sin(dLat/2) +
                Math.cos(lat1*Math.PI/180) *
                Math.cos(lat2*Math.PI/180) *
                Math.sin(dLon/2) * Math.sin(dLon/2);

            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }

        navigator.geolocation.getCurrentPosition(function(position){

            var lat = position.coords.latitude;
            var lon = position.coords.longitude;

            var service = new google.maps.places.PlacesService(document.createElement('div'));

            service.nearbySearch({
                location: {lat: lat, lng: lon},
                radius: 5000,
                keyword: "pulmonologist hospital"
            }, function(results, status){

                if(status === google.maps.places.PlacesServiceStatus.OK){

                    var html = "";

                    for(var i=0;i<results.length;i++){

                        var place = results[i];

                        var dist = getDistance(
                            lat,
                            lon,
                            place.geometry.location.lat(),
                            place.geometry.location.lng()
                        ).toFixed(2);

                        html += "<b>"+place.name+"</b><br>";
                        html += "⭐ Rating: "+place.rating+"<br>";
                        html += "📍 Distance: "+dist+" km<br><br>";

                    }

                    document.getElementById("results").innerHTML = html;

                }

            });

        });

        </script>

        <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyCbxuqZwVoUx7ItP-HsPY-bXvk8V3Q7ZGE&libraries=places"></script>

        """, height=500)
