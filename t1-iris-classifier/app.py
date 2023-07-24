import streamlit as st
from source.pipelines.testing_pipeline import custom_data,prediction_pipeline 
st.title("Iris species classification using Machine Learning ")


st.image("static/main.jpeg", use_column_width=True)




st.write("""
    ---\n
    Made with ❤️ by Chakka Guna Sekhar Venkata Chennaiah.
    """)

st.subheader('Enter Form')
sepal_length = st.number_input('Enter the Sepal Length')
sepal_width = st.number_input('Enter the Sepal Width')
petal_length = st.number_input('Enter the Petal Length')
petal_width= st.number_input('Enter the Petal Width')

data=custom_data(sepal_length,sepal_width,petal_length,petal_width)
pred_df=data.get_data_as_a_dataframe()

pred_pipe=prediction_pipeline()
final_pred=pred_pipe.predict(pred_df)

if st.button('Predict'):
    if final_pred[0]==0:
        st.write("Based on the data, the most likely species of flower is Iris Setosa.")
        st.image('static/iris-setosa.png',use_column_width=True)
    elif final_pred[0]==1:
        st.write("Based on the data, the most likely species of flower is Iris Versicolor.")
        st.image('static/iris-versicolor.jpeg',use_column_width=True)
    elif final_pred[0]==2:
        st.write("Based on the data, the most likely species of flower is Iris Virginica.")
        st.image('static/iris-virginica.png',use_column_width=True)
