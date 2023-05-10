import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://static.vecteezy.com/system/resources/thumbnails/012/010/273/original/heart-beat-ecg-medical-background-loop-animation-heart-rhythm-ekg-background-cardiogram-in-heart-shape-heart-pulse-neon-glowing-ecg-neon-heart-beat-rhythm-motion-background-heart-beat-pulse-neon-free-video.jpg");
    background-size: 120%;
    background-position: top left;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """

st.markdown(page_bg_img, unsafe_allow_html=True)




with st.sidebar:
    st.image("heartfly.gif")
    st.image("sty8.gif")
    pass

# Insert containers separated into tabs:
tab1, tab2 , tab3 , tab4 , tab5= st.tabs(["Artifact", "Extrahls","Murmur","Normal","Extrasytole"])

#Artifact
with tab1:
    #st.radio('Select one:', [1, 2])
    st.write("Artifact")

    with st.expander("More about Artifact"):
        st.markdown(
        '''<html>
        <head>
        <style>
        body {
        font-family: Georgia, serif;
        color:blue;
        background-color:powderblue;
        }
        </style>
        </head>
        <body>
            In the Artifact category there are a wide range of different sounds, 
        including feedback squeals and echoes, speech, music and noise. 
        There are usually no discernable heart sounds, and thus little or no temporal 
        periodicity at frequencies below 195 Hz. This category is the most different from the others.
        It is important to be able to distinguish this category from the other three categories, 
        so that someone gathering the data can be instructed to try again.

        ''',unsafe_allow_html=True)


    st.subheader("Artifact - Audio")
    st.audio(r"E:\DAVL\Package\Data\set_a\artifact__201105040918.wav")


    artifact_file = r"E:\DAVL\Package\Data\set_a\artifact__201105040918.wav"
    y2, sr2 = librosa.load(artifact_file,duration=5)
    dur=librosa.get_duration(y=y2)

    st.write(y2.shape,sr2)


    # show it
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y2, sr=sr2)
    plt.title("Artifact - Normalised")

    st.area_chart(y2)
    if st.button("More plot"):
        st.image("Extrahls.png")

    #st.line_chart(y2)

    #st.snow()

#Extrahls
with tab2:
    #st.radio('Select one:', [1, 2])
    st.write("Extrahls")

    with st.expander("More about Extrahls"):
        st.markdown(
        '''<html>
        <head>
        <style>
        body {
        font-family: Georgia, serif;
        color:blue;
        background-color:powderblue;
        }
        </style>
        </head>
        <body>
            

        ''',unsafe_allow_html=True)


    st.subheader("Extrahls- Audio")
    st.audio(r"Data\set_a\extrahls__201101070953.wav")


    artifact_file = r"Data\set_a\extrahls__201101070953.wav"
    y2, sr2 = librosa.load(artifact_file,duration=5)
    dur=librosa.get_duration(y=y2)

    st.write(y2.shape,sr2)


    # show it

    st.area_chart(y2)
    if st.button("More-plot"):
        st.image("Extrahls.png")

    #st.line_chart(y2)

    #st.snow()

#Murmur
with tab3:
    #st.radio('Select one:', [1, 2])
    st.write("Murmur")

    with st.expander("More about Murmur"):
        st.markdown(
        '''<html>
        <head>
        <style>
        body {
        font-family: Georgia, serif;
        color:blue;
        background-color:powderblue;
        }
        </style>
        </head>
        <body>
            Murmur Category
            Heart murmurs sound as though there is a “whooshing, roaring, rumbling, or turbulent fluid” 
            noise in one of two temporal locations: (1) between “lub” and “dub”, or (2) between “dub” and “lub”.
            They can be a symptom of many heart disorders, some serious. There will still be a “lub” and a “dub”. 
            One of the things that confuses non-medically trained people is that murmurs happen between lub and dub 
            or between dub and lub; not on lub and not on dub. Below, you can find an asterisk* at the locations a murmur may be.
            …lub..****...dub……………. lub..****..dub ……………. lub..****..dub ……………. lub..****..dub … or
            …lub……….dub…******….lub………. dub…******….lub ………. dub…******….lub ……….dub…

            Dataset B also contains noisy_murmur data - murmur data which includes a substantial amount of background noise 
            or distortion. You may choose to use this or ignore it, however the test set will include some equally noisy example

        ''',unsafe_allow_html=True)


    st.subheader("Murmur - Audio")
    a_file = r"Data\set_a\murmur__201101051104.wav"
    st.audio(a_file)
    
    y2, sr2 = librosa.load(artifact_file,duration=5)
    dur=librosa.get_duration(y=y2)

    st.write(y2.shape,sr2)


    # show it
    st.area_chart(y2)
    if st.button("More_plot"):
        st.image("Murmur.png")

    #st.line_chart(y2)

    #st.snow()

#Normal
with tab4:
    #st.radio('Select one:', [1, 2])
    st.write("Normal")

    with st.expander("More about Normal"):
        st.markdown(
        '''<html>
        <head>
        <style>
        body {
        font-family: Georgia, serif;
        color:blue;
        background-color:powderblue;
        }
        </style>
        </head>
        <body>
        Normal Category
        In the Normal category there are normal, healthy heart sounds. These may contain noise in the final second
        of the recording as the device is removed from the body. They may contain a variety of background noises 
        (from traffic to radios). They may also contain occasional random noise corresponding to breathing, or
         brushing the microphone against clothing or skin. A normal heart sound has a clear “lub dub, lub dub” pattern,
        with the time from “lub” to “dub” shorter than the time from “dub” to the next “lub” 
        (when the heart rate is less than 140 beats per minute). Note the temporal description 
        of “lub” and “dub” locations over time in the following illustration:

        …lub……….dub……………. lub……….dub……………. lub……….dub……………. lub……….dub…

        In medicine we call the lub sound "S1" and the dub sound "S2".
        Most normal heart rates at rest will be between about 60 and 100 beats (‘lub dub’s) per minute. 
        However, note that since the data may have been collected from children or adults in calm or excited states,
        the heart rates in the data may vary from 40 to 140 beats or higher per minute. Dataset B also contains noisy_normal
        data - normal data which includes a substantial amount of background noise or distortion. You may choose to use this or 
        ignore it, however the test set will include some equally noisy examples.


        ''',unsafe_allow_html=True)


    st.subheader("Normal - Audio")
    


    a_file = r"Data\set_a\normal__201101070538.wav"
    st.audio(a_file)
    y2, sr2 = librosa.load(a_file,duration=5)
    dur=librosa.get_duration(y=y2)

    st.write(y2.shape,sr2)


    # show it
    

    st.area_chart(y2)
    if st.button("More plot.."):
        st.image("Normal.png")

    #st.line_chart(y2)

    #st.snow()

#
with tab5:
    #st.radio('Select one:', [1, 2])
    st.write("Extrasytole")

    with st.expander("More about Extrasytole"):
        st.markdown(
        '''<html>
        <head>
        <style>
        body {
        font-family: Georgia, serif;
        color:blue;
        background-color:powderblue;
        }
        </style>
        </head>
        <body>
            Extrasystole sounds may appear occasionally and can be identified because there is a heart 
            sound that is out of rhythm involving extra or skipped heartbeats, e.g. a “lub-lub dub” or a “lub dub-dub”. 
            (This is not the same as an extra heart sound as the event is not regularly occuring.) 
            An extrasystole may not be a sign of disease. It can happen normally in an adult and can be very 
            common in children. However, in some situations extrasystoles can be caused by heart diseases.
            If these diseases are detected earlier, then treatment is likely to be more effective. 
            Below, note the temporal description of the extra heart sounds:
            …........lub……….dub………..………. lub. ………..……….dub…………….lub.lub……..…….dub…….
            or
            …lub………. dub......………………….lub.…………………dub.dub………………….lub……..…….dub.…

        ''',unsafe_allow_html=True)


    st.subheader("Extrasyole - Audio")
    
    a_file = r"Data\set_b\extrastole__127_1306764300147_C2.wav"
    st.audio(a_file)
    y2, sr2 = librosa.load(artifact_file,duration=5)
    dur=librosa.get_duration(y=y2)

    st.write(y2.shape,sr2)

    st.area_chart(y2)
    if st.button("More.plot"):
        st.image("Extrasytole.png")

    #st.line_chart(y2)

    #st.snow()



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