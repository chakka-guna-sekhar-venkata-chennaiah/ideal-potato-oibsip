import streamlit as st
from source.pipelines.testing_pipeline import prediction_pipeline 
st.title("Email spam classifier using Machine Learning ")


st.image("static/main.png", use_column_width=True)




st.write("""
    ---\n
    Made with ❤️ by Chakka Guna Sekhar Venkata Chennaiah.
    """)

input_sms = st.text_area("Enter the message:- ")



pred_pipe=prediction_pipeline()
final_pred=pred_pipe.predict(input_sms)

if st.button('Predict'):
    if final_pred==1:
        st.write("According to the content of your message, it appears to be categorized as spam.")
        st.image('static/spam.png',use_column_width=True)
    elif final_pred==0:
        st.write("Upon reviewing the contents of your message, it is determined that your email is not classified as spam.")
        st.image('static/not_spam.png',use_column_width=True)
