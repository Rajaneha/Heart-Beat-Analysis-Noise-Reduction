import streamlit as st
import pandas as pd

page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://img.lovepik.com/background/20211022/large/lovepik-fluid-gradient-background-image_401733169.jpg");
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

df = pd.read_csv("power bi\inpatientCharges.csv\inpatientCharges.csv")
st.subheader("Description")
st.write(df.head())
qw = df['DRG Definition'].unique()



c1 , c2 = st.columns(2)

with c1:
    st.subheader("ALL types of diseases ..>")
    st.write(qw)
with c2:
    hd = []
    st.subheader("Heart Diseases")
    for i in qw:
        j = i.lower()
        if ("cardi" in j or "heart" in j):
            hd.append(i)
    st.write(pd.DataFrame(hd))

st.write("Proportion of heart disease to All type of disease"+str(len(hd)/len(qw)))


uq = df['Provider City'].unique()


#add heart diesease prediction
st.title("Check for Heart diesase :")
l = []
st.markdown("[Click here to go to code!](https://colab.research.google.com/drive/1WjuNtY4aqvcUuHieRjA0rgxH7AY3C2_P#scrollTo=rhE30gDTarKD)")

l.append(st.number_input("Enter params AGE:"))
l.append(st.number_input("Enter params SEX:"))
l.append(st.number_input("Enter params CP- DISEASE:"))
l.append(st.number_input("Enter params TRESTBPS:"))
l.append(st.number_input("Enter params CHOLESTROL:"))
l.append(st.number_input("Enter params FBS:"))
l.append(st.number_input("Enter params RESTECG:"))
l.append(st.number_input("Enter params THALACH:"))
l.append(st.number_input("Enter params EXANG:"))
l.append(st.number_input("Enter params OLDPEAK:"))
l.append(st.number_input("Enter params SLOPE:"))
l.append(st.number_input("Enter params CA:"))
l.append(st.number_input("Enter params THAL:"))
if st.button("Calculate"):
    import pandas as pd
    df_copy=pd.read_csv('heart.csv')

    df_copy.target=df_copy.target.map({1:'Yes', 0:'No'})
    df_copy.sex= df_copy.sex.map({1:'Male',0:'Female'})
    df_copy.cp= df_copy.cp.map({0:'Normal',1:'Atypical Angina',2:'Non Anginal Pain', 3:'Asymptotic'})
    df_copy.head()

    # Commented out IPython magic to ensure Python compatibility.
    import matplotlib.pyplot as plt


    df=pd.read_csv('heart.csv')
    x= df.drop(['target'], axis=1)
    y = df['target']
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=40)
    model = LogisticRegression()
    model.fit(X_train,y_train)

    prediction= model.predict(X_test)
    from sklearn.metrics import confusion_matrix

    confusion_mtx=confusion_matrix(y_test,prediction)
    import numpy as np

    v = np.array([l])
    val = model.predict(v)

    if val == 1:
        st.success("You're free from heart diesease")
    else:
        st.error("I think you have some problems related to heart")


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