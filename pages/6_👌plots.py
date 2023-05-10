import streamlit as st

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://media.istockphoto.com/id/1347345879/photo/financial-rising-graph-and-chart-with-lines-and-numbers.jpg?b=1&s=170667a&w=0&k=20&c=Pi9eTsg9y20nRY1N-lnRDZqKemW6bXpkeHh2zrbauuA=");
background-size: 100%;
background-position: top left;
background-attachment: local;
}}
a-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""


st.markdown(page_bg_img,unsafe_allow_html=True)

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
		Is your name Google? Because you have everything I’ve been searching for.
	</marquee>
</body>
</html>
""",unsafe_allow_html=True)



sbar = st.sidebar
with sbar:
    pass
    st.image("heartfly.gif")
    st.image("sty9.gif")
    #st.markdown('<img src="https://thumbs.dreamstime.com/z/bundle-charts-diagrams-schemes-graphs-plots-various-types-statistical-data-financial-information-bundle-charts-157487481.jpg" style="width:100px; height:900px">',unsafe_allow_html=True)


url = "https://rajaneha.github.io/graphs/"#html code link to be pasted

# Define the HTML for the clickable image
html = f'<a href="{url}" target="_blank"><img src="https://thumbs.dreamstime.com/z/bundle-charts-diagrams-schemes-graphs-plots-various-types-statistical-data-financial-information-bundle-charts-157487481.jpg" style="width:200px; height:200px"> </a>'

# Display the clickable image using st.markdown()

st.markdown(html, unsafe_allow_html=True)
st.image("tap.gif")



#add heart rate calculator and others


#
#v = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
#print(model.predict(v))


def calculate_heart_rate(age):
    return 220 - age

st.title("Heart Rate Calculator")

# Get user input for age
age = st.number_input("Enter your age", min_value=1)

# Calculate heart rate based on age
heart_rate = calculate_heart_rate(age)

# Display heart rate to user
st.success(f"Your estimated maximum heart rate is {heart_rate} beats per minute.")

def calculate_hrr(age, resting_hr):
    max_hr = 220 - age
    hrr = max_hr - resting_hr
    return hrr

st.title("Heart Rate Reserve Calculator")
with st.expander("HRR"):
    st.write("""Heart rate reserve (HRR) is a measure of the difference between your resting heart rate and your maximum heart rate. It represents the range of heartbeats that your heart can produce in response to physical activity.""")

# Get user input for age and resting heart rate
age = st.number_input("Enter your age :", min_value=1)
resting_hr = st.number_input("Enter your resting heart rate", min_value=1)

# Calculate heart rate reserve based on user input
hrr = calculate_hrr(age, resting_hr)

# Display heart rate reserve to user
st.success(f"Your heart rate reserve is {hrr} beats per minute.")

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