FROM python:3.7
WORKDIR /sales-prediction
COPY . .
RUN python3 -m pip install -r requirements.txt
RUN python3 source/components/data_loading.py
EXPOSE 9998
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"] 
