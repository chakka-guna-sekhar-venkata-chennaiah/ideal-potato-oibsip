FROM python:3.7
WORKDIR /iris-classification
COPY . .
RUN python3 -m pip install -r requirements.txt
RUN python3 source/components/data_loading.py
EXPOSE 9999
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"] 
