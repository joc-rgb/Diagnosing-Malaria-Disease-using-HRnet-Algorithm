from fastai.vision.widgets import *
from fastai.vision.all import *

from pathlib import Path

import streamlit as st

st.set_page_config(layout="centered", page_icon="🩸", page_title="Malaria Detection System")
st.title("🩸 Malaria Detection System")
st.image("https://www.news-medical.net/images/Article_Images/ImageForArticle_22209_16467383903163122.jpg", caption=None)
st.caption("Malaria is a disease caused by Plasmodium parasites that remains a major threat in global health, affecting 200 million people and causing 400,000 deaths a year. The main species of malaria that affect humans are Plasmodium falciparum and Plasmodium vivax.")
st.subheader(
    ":blue[Are you at risk of Malaria disease?]"
)
st.info("Always seek the guidance of your doctor or other qualified health professional with any questions you may have regarding your health or a medical condition",icon="ℹ️")
st.write("##")
st.write("##")
class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(Path()/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            st.write("##")
            
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'**Prediction: {pred}; Probability: {probs[pred_idx]:.04f}**')
            if pred=="uninfected":
                st.markdown("**YOUR RISK FOR MALARIA DISEASE IS **:green[VERY MINIMAL]** AT THIS MOMENT**")
            else:
                st.markdown("**YOUR RISK FOR MALARIA DISEASE IS **:red[VERY HIGH]** AT THIS MOMENT.**") 
                st.markdown("**Please consult a professional doctor :orange[as soon as possible] for clarification and treatment.**")
                            
                
            
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':

    file_name='malaria.pkl'

    predictor = Predict(file_name)
