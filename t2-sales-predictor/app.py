import streamlit as st
from source.pipelines.testing_pipeline import custom_data,prediction_pipeline 
st.title("Sales prediction of a company based on Advertisement data")


st.image("static/main.png", use_column_width=True)




st.write("""
    ---\n
    Made with ❤️ by Chakka Guna Sekhar Venkata Chennaiah.
    """)

st.subheader('Enter Form')
tv = st.number_input('Enter the amount you spend for the tv advertisement')
radio = st.number_input('Enter the amount you spend for the radio advertisement')
newspaper = st.number_input('Enter the amount you spend for the newspaper advertisement')

data=custom_data(tv,radio,newspaper)
pred_df=data.get_data_as_a_dataframe()

pred_pipe=prediction_pipeline()
final_pred=pred_pipe.predict(pred_df)

if st.button('Predict'):
    fpred=final_pred[0]
    st.write("Based on the data, the sales of the company will be {}".format(fpred))
    
    