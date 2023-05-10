import streamlit as st


st.markdown("""
<html>
<head>
	<title>Marquee Example</title>
	<style>
		/* CSS styles */
		.marquee {
			font-size: 24px; /* change to desired font size */
            color:red;
            font-family: sans-serif;
		}
	</style>
</head>
<body>
	<marquee behavior="scroll" direction="left" class="marquee" scrollamount="10">
		NO BEAUTY SHINES THAN A BETTER HEART
	</marquee>
</body>
</html>
""",unsafe_allow_html=True)




st.markdown("[Click here to go to Power BI!](https://app.powerbi.com/reportEmbed?reportId=09ab8da6-bf1e-467a-bd9b-1f259be3b5d4&autoAuth=true&ctid=bd01fe4c-cfb9-4e3a-beee-cd2171d6ab38)")



st.write("""<iframe title="start" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=09ab8da6-bf1e-467a-bd9b-1f259be3b5d4&autoAuth=true&ctid=bd01fe4c-cfb9-4e3a-beee-cd2171d6ab38" frameborder="0" allowFullScreen="true"></iframe>"""
         ,unsafe_allow_html=True)

import folium

from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

geolocator = Nominatim(user_agent="my-app")

# center on Liberty Bell, add marker
m = folium.Map(location=[39.949610, -75.150282], zoom_start=3)
file = open("mapplot.csv",'r')

for i in range(4):
    a = file.readline()
    l = a.split("=")

    folium.Marker(
        [float(l[0]), float(l[1])], popup="Liberty Bell", tooltip="Liberty Bell"
    ).add_to(m)

# call to render Folium map in Streamlit
st_data = st_folium(m, width=500,height=400)
st.markdown(
    """
    <div class="icon-container">
      <a href="#" onclick="showAccount('instagram')"><img src="https://cdn4.iconfinder.com/data/icons/social-media-icons-the-circle-set/48/instagram_circle-512.png" width="30" height="30"></a>
      <a href="#" onclick="showAccount('whatsapp')"><img src="https://cdn4.iconfinder.com/data/icons/social-messaging-ui-color-shapes-2-free/128/social-whatsapp-circle-512.png" width="30" height="30"></a>
      <a href="#" onclick="showAccount('twitter')"><img src="https://cdn3.iconfinder.com/data/icons/social-icons-5/607/Twitterbird.png" width="30" height="30"></a>
      <a href="#" onclick="showAccount('mail')"><img src="https://cdn2.iconfinder.com/data/icons/social-icons-circular-color/512/gmail-512.png" width="30" height="30"></a>
    </div>
    """,
    unsafe_allow_html=True)
