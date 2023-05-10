import streamlit as st

with st.sidebar:
    st.image("heartfly.gif")
    st.image("sty6.gif")
    pass

page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://c4.wallpaperflare.com/wallpaper/764/505/66/baby-groot-4k-hd-superheroes-wallpaper-preview.jpg");
    background-size: 100%;
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

st.header("Dataset Description")
c1,c2,c3 = st.tabs(["Dataset-A","Dataset-B","Organisation"])

with c1:
    st.subheader("Dataset - A")
    st.code("""
        Dataset A, containing 176 files in WAV format, organized as:
        Atraining_normal.zip	14Mb	31 files	
        Atraining_murmur.zip	17.3Mb	34 files	
        Atraining_extrahs.zip	6.9Mb	19 files	
        Atraining_artifact.zip	22.5Mb	40 files	
        Aunlabelledtest.zip	    24.6Mb	52 files	
        The same datasets are also available in aif format:

        Atraining_normal.zip	13.2Mb	31 files	
        Atraining_murmur.zip	16.4Mb	34 files	
        Atraining_extrahs.zip	6.5Mb	19 files	
        Atraining_artifact.zip	20.9Mb	40 files	
        Aunlabelledtest.zip	    23.0Mb	52 files	""")
    
    st.markdown("[Click here to download](http://www.peterjbentley.com/heartchallenge/)")

with c2:
    st.subheader("Dataset - B")
    st.code("""
        Dataset B, containing 656 files in WAV format, organized as:

        Btraining_normal.zip (containing sub directory Btraining_noisynormal)	13.8Mb	320 files	
        Btraining_murmur.zip (containing subdirectory Btraining_noisymurmur)	5.3Mb	95 files	
        Btraining_extrasystole.zip	                                            1.9Mb	46 files
        Bunlabelledtest.zip	                                                    9.2Mb	195 files	
        The same datasets are also available in aif format:

        Btraining_normal.zip (containing sub directory Btraining_noisynormal)	13.0Mb	320 files	
        Btraining_murmur.zip (containing subdirectory Btraining_noisymurmur)	5.1Mb	95 files	
        Btraining_extrasystole.zip	                                            2.1Mb	46 files	
        Bunlabelledtest.zip	                                                    8.7Mb	195 files
        """)
    
    st.markdown("[Click here to download](http://www.peterjbentley.com/heartchallenge/)")
with c3:
    st.code("""	
Data Description and Organisation
Please use the following citation if the data is used:

@misc{pascal-chsc-2011,
author = "Bentley, P. and Nordehn, G. and Coimbra, M. and Mannor, S.",
title = "The {PASCAL} {C}lassifying {H}eart {S}ounds {C}hallenge 2011 {(CHSC2011)} {R}esults",
howpublished = "http://www.peterjbentley.com/heartchallenge/index.html"}
The audio files are of varying lengths, between 1 second and 30 seconds (some have been clipped 
to reduce excessive noise and provide the salient fragment of the sound).
Most information in heart sounds is contained in the low frequency components, with noise in the
higher frequencies. It is common to apply a low-pass filter at 195 Hz. Fast Fourier transforms are 
also likely to provide useful information about volume and frequency over time. More domain-specific
knowledge about the difference between the categories of sounds is provided below.""")

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