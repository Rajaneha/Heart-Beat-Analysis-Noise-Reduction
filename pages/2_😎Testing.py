import streamlit as st
import os
import numpy as np
import os
import time
import wave
import struct
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import pyplot as plt
from scipy import fftpack
from io import BytesIO

def reduce(audiofile):
  lt = []
  wav_file = wave.open(audiofile, 'r')
  nchannels, sampwidth, framerate, nframes, comptype, compname = wav_file.getparams()
  lt.append("Params:"+"\n\tChannel:"+ str(nchannels)+"\n\tSample Width:"+str(sampwidth)+ "\n\tFramerate:"+str(framerate)+"\n\tNumber of Frames:"+str(nframes)+str( "\n\tcomptype:")+str( comptype)+str( "\n\tCompname:")+ str(compname))

  # Reading wave format data from wav file.
  frames_wave = wav_file.readframes(nframes)
  wav_file.close()

  lt.append("Length:"+str( nframes))

  # Deserializing
  frames_wave = struct.unpack('{n}h'.format(n=nframes), frames_wave)
  frames_wave = np.array(frames_wave)
  lt.append("Min value:"+str(np.min(frames_wave))+ "Max value:" +str(np.max(frames_wave)))


  frames_freq_domian = np.fft.fft(frames_wave)
  rames_freq_domian = fftpack.fft(frames_wave)

  # Above value is in complex number but we want absolute number
  # This will give us the frequency we want
  magnitude = np.abs(frames_freq_domian)  # Or ampliude ?
  phase = np.angle(frames_freq_domian) # Normally we are not interested in phase information, its only used in reconstruction.

  lt.append(str(magnitude.shape)+str(phase.shape))
  lt.append("The max frequency (highest magnitude) is {} Hz".format(np.where(magnitude == np.max(magnitude))[0][0]))


  fig = plt.figure(figsize = (25, 6))
  fig.suptitle('Original wav data')

  ax1 = fig.add_subplot(1,3,1)
  ax1.set_title("Original audio wave / Spatial Domain")
  ax1.set_xlabel("Time(s)")
  ax1.set_ylabel("Amplitude (16 bit depth - Calulated above)")

  ax1.plot(frames_wave)

  ax2 = fig.add_subplot(1,3,2)
  ax2.set_title("Frequency by magnitude (Max at {} Hz) / Frequency Domain".format(np.where(magnitude == np.max(magnitude))[0][0]))
  ax2.set_xlabel("Frequency (Hertz)")
  ax2.set_ylabel("Magnitude (normalized)")
  ax2.set_xlim(0, 44100)  # we are not interested in rest
  ax2.plot(magnitude / nframes)  # Normalizing magnitude

  ax3 = fig.add_subplot(1,3,3)
  ax3.set_title("[Unclipped]Frequency by magnitude (Max at {} Hz) / Frequency Domain".format(np.where(magnitude == np.max(magnitude))[0][0]))
  ax3.set_xlabel("Frequency (Hertz)")
  ax3.set_ylabel("Magnitude (normalized)")
  ax3.plot(magnitude / nframes)  # Normalizing magnitude

  #-->plt.show()
  plt.savefig("img1")
      


  def butter_pass_filter(data, cutoff, fs, order=5):
      nyq = 0.5 * fs # Nyquist frequency
      normal_cutoff = cutoff / nyq  # A fraction b/w 0 and 1 of sampling rate
      lt.append("normal_cutoff:"+str(normal_cutoff)+str( (data.shape[0] / 2) * normal_cutoff)) # Tricky ? 
      b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
      y = signal.filtfilt(b, a, data)

      def _plot_graph():
        # Get the filter coefficients so we can check its frequency response.
        # Plot the frequency response.
        w, h = signal.freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5 *fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*fs)
        plt.title("Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        #-->plt.grid()
        #-->plt.show()
        plt.savefig("img2.png")
      _plot_graph()
      return y

  # Filter requirements.
  order = 10
  fs = framerate #* 6.28  # sample rate, Hz
  cutoff =  900 #* 6.28      # desired cutoff frequency of the filter, Hz

  # Get the filter coefficients so we can check its frequency response.
  y = butter_pass_filter(frames_wave, cutoff, fs, order)

  print(frames_wave.shape, y.shape, np.array_equal(frames_wave, y))
  fig = plt.figure(figsize = (25, 6))
  # fig.suptitle('Horizontally stacked subplots')

  ax1 = fig.add_subplot(1,4,1)
  ax1.set_title("[After Filter] Original audio wave / Spatial Domain")
  ax1.set_xlabel("Time(s)")
  ax1.set_ylabel("Amplitude (16 bit depth - Calulated above)")
  ax1.plot(y)

  ax2 = fig.add_subplot(1,4,2)
  ax2.set_title("[Before Filter] Original audio wave / Spatial Domain")
  ax2.set_xlabel("Time(s)")
  ax2.set_ylabel("Amplitude (16 bit depth - Calulated above)")
  ax2.plot(frames_wave, 'r')


  m = np.abs(fftpack.fft(y))
  ax3 = fig.add_subplot(1,4,3)
  ax3.set_title("[After Filter] Frequency by magnitude")
  ax3.set_xlabel("Frequency (Hertz)")
  ax3.set_ylabel("Magnitude (normalized)")
  ax3.set_xlim(0, 44100)  # we are not interested in rest
  ax3.plot(np.abs(fftpack.fft(y)) / nframes)
  # ax2.plot(range(0, 676864), m, 'g-', label='dataa')


  ax4 = fig.add_subplot(1,4,4)
  ax4.set_title("[Before Filter] Frequency by magnitude")
  ax4.set_xlabel("Frequency (Hertz)")
  ax4.set_ylabel("Magnitude (normalized)")
  ax4.set_xlim(0, 44100)  # we are not interested in rest
  # ax2.plot(magnitude * 2 / (16 * len(magnitude)))
  ax4.plot(magnitude / nframes, 'r')

  st.pyplot(fig)
  plt.savefig("img3.png")

  amplitude = 1
  filtered_file = "filteredapi/"+audiofile.name
  wav_file = wave.open(filtered_file, 'w')
  # The tuple should be (nchannels, sampwidth, framerate, nframes, comptype, compname)
  wav_file.setparams((nchannels, sampwidth, framerate, nframes, "NONE", "not compressed"))

  #Struct is a Python library that takes our data and packs it as binary data. The h in the code means 16 bit number.
  for s in y:
      wav_file.writeframes(struct.pack('h', int(s*amplitude)))

  wav_file.close()
  return lt , filtered_file


with st.sidebar:
    st.image("heartfly.gif")
    st.image("sty5.gif")

page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://wallpapers.com/images/featured/ph3fw6k03ddbmbmh.jpg");
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

#st.markdown(page_bg_img, unsafe_allow_html=True)


t1 , t2 , t3 , t4 = st.tabs(['Testing - Known','Testing - UnKnown','Noise Reduction','Label the unlabelled'])
import os 
import streamlit as st
import fnmatch

import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from PIL import Image
img = Image.open("loading.gif")

# load the audio file and its labels

model = load_model('best_model_trained.hdf5')



def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

def strpredict(filename):
    prediction_feature=features_extractor(filename)
    prediction_feature=prediction_feature.reshape(1,-1)
    tr = model.predict(prediction_feature)
    keyoflabel = np.argmax(tr)
    return keyoflabel

CLASSES = ['artifact','extrahls','murmur','normal']

inp = "Data"

l = os.listdir(inp+'/set_b')
e = os.listdir(inp+'/set_a')

A_a = fnmatch.filter(os.listdir(inp+'/set_a'), 'artifact*.wav')
A_n = fnmatch.filter(os.listdir(inp+'/set_a'), 'normal*.wav')
A_e = fnmatch.filter(os.listdir(inp+'/set_a'), 'extrahls*.wav')
A_m = fnmatch.filter(os.listdir(inp+'/set_a'), 'murmur*.wav')



unlab = []
known = A_a + A_e + A_m + A_n
noisy = []


l = e+l
for i in l:
    if 'unlabel' in i:
        unlab.append(i)
    if 'noisy' in i:
        noisy.append(i)




with t1:
    #labelled
    st.header("Testing with known Data")
    col1 , col2 = st.columns(2)
    
    with col1:
        
        filek = st.selectbox("Select an wav file--)",known)
        if st.button("Test Known"):
            st.success(CLASSES[strpredict(inp+'/set_a/'+filek)])
    with col2:
        up = st.file_uploader("Choose a file-")
        if up != None:
            st.success(CLASSES[strpredict(up)])

with t2:
    st.header("Testing with Unknown Data")
    col1 , col2 = st.columns(2)
    with col1:
        
        filek = st.selectbox("Select an wav file--)",unlab)
        if st.button("Test Known."):
            st.success(CLASSES[strpredict(inp+'/set_a/'+filek)])
    with col2:
        up2 = st.file_uploader("Choose a file")
        if up2 != None:
            st.success(CLASSES[strpredict(up2)])
with t3:
    progress_bar = st.progress(0)

    # Simulate some action
    for i in range(100):
        progress_bar.progress(i + 1)
        
    background_color = 'black'

    
    vgg = st.file_uploader("Enter the noisy")
    if vgg != None:
        try:
            ww = reduce(vgg)
            st.header("Noise reduced ;)")
            st.balloons()

            lt = ww[0]
            ms = ww[1]
            d=''
            for i in lt:
                d += i 
                st.write(f'<p style="background-color:{background_color}">{i}</p>', unsafe_allow_html=True)
            st.audio(ms)
            st.download_button("Download Reduced file",ms)

            from fpdf import FPDF

            # Create a PDF object
            pdf = FPDF()

            # Add a page
            pdf.add_page()

            # Set the font and size for the title
            pdf.set_font("Arial", "B", 16)

            # Write the title
            pdf.cell(0, 10, "PDF Report - Noise reduction", 1, 1, "C")
            pdf.multi_cell(0, 10, d)

            # Add the images
            pdf.image("img1.png", x=10, y=100, w=200)
            pdf.image("img2.png", x=10, y=200, w=200)

            pdf.add_page()

            pdf.image("img3.png", x=10, y=30, w=200)

            # Save the PDF
            pdf.output("my_pdf_with_images.pdf")


            # Create a download button for the PDF file
            with open("my_pdf_with_images.pdf", "rb") as f:
                pdf_bytes = f.read()
            st.download_button(label="Download PDF", data=pdf_bytes, file_name="my_pdf_file.pdf", mime="application/pdf")


        except:
            st.warning("Frames out of range")

with t4:
    st.image("sty11.gif")
    opt = st.multiselect("Dump files to be labelled",unlab)
    if st.button("Test knowN"):
      for i in opt:
            print(i)
            if 'Bunlab' in i:
                st.success(i+"   "+CLASSES[strpredict(inp+'/set_b/'+i)])
            else:
                st.success(i+"   "+CLASSES[strpredict(inp+'/set_a/'+i)])

    
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