
# Iris Classification System using MachineLearning 🐍
<div align='center'>
<img src='https://github.com/software-babooi/ideal-potato-oibsip/assets/110555361/990476f8-9c47-4938-9889-87811658df1e'>
</div>
<br>

## 1. :hammer_and_wrench: Requirements
<div align='center'>
<img alt="Static Badge" src="https://img.shields.io/badge/2.8-green?style=for-the-badge&logo=python&logoColor=%233776AB&label=Python&labelColor=%23FFFFFF&color=%23008000">&nbsp
<img alt="Static Badge" src="https://img.shields.io/badge/2.8-green?style=for-the-badge&logo=tensorflow&logoColor=%23FF6F00&label=Tensorflow&labelColor=%23FFFFFF&color=%23008000">&nbsp
<img alt="Static Badge" src="https://img.shields.io/badge/1.22.0-green?style=for-the-badge&logo=streamlit&logoColor=%23FF4B4B&label=Streamlit&labelColor=%23FFFFFF&color=%23008000">&nbsp
<img alt="Static Badge" src="https://img.shields.io/badge/2.12.0-green?style=for-the-badge&logo=keras&logoColor=%23D00000&label=keras&labelColor=%23FFFFFF&color=%23008000">&nbsp
<img alt="Static Badge" src="https://img.shields.io/badge/1.2.2-green?style=for-the-badge&logo=scikit-learn&logoColor=%23F7931E&label=scikit-learn&labelColor=%23FFFFFF&color=%23008000">&nbsp
<img alt="Static Badge" src="https://img.shields.io/badge/1.23.5-green?style=for-the-badge&logo=numpy&logoColor=%23013243&label=numpy&labelColor=%23FFFFFF&color=%23008000">&nbsp
<img alt="Static Badge" src="https://img.shields.io/badge/1.5.2-green?style=for-the-badge&logo=pandas&logoColor=%23150458&label=pandas&labelColor=%23FFFFFF&color=%23008000">




</div>


<br>

## 2. 📝 __Description__
The __Iris Classifier System Using MachineLearning__ is a software project that employs advanced deep learning algorithms to accurately classify iris flowers based on their species. It is designed for researchers, botanists, and enthusiasts interested in iris flower classification and taxonomy.

The repository is organized in a modular fashion, adhering to industry-level coding standards. Working with this repository ensures a professional and structured approach to code development. 


## 3. 🌲 __Project Directory Structure__  
```bash
├── Dockerfile
├── README.md
├── app.py
├── notebook
│   └── IRIS_1.csv 📖
├── requirements.txt
├── setup.py
├── source
│   ├── __init__.py
│   ├── components
│   │   ├── __init__.py
│   │   ├── data_loading.py
│   │   ├── keras_tuning.py
│   │   ├── model_training.py
│   │   └── transformation.py
│   ├── exception.py
│   ├── loggers.py
│   ├── pipelines
│   │   ├── __init__.py
│   │   └── testing_pipeline.py
│   └── utlis.py
└── static
    ├── iris-setosa.png
    ├── iris-versicolor.jpeg
    ├── iris-virginica.png
    └── main.jpeg

```
1. __Dockerfile__: This file contains the instructions to build a Docker image for your project, facilitating easy deployment and reproducibility.
2. __README.md__: The README file provides essential information about the project, including its description, usage instructions, and deployment details.
3. __app.py__: This file contains the code for the user interface of your __Iris Classifier System__.
4. __notebook__: This directory stores the notebook containing the dataset IRIS_1.csv.
5. __requirements.txt__: The requirements file lists all the dependencies and their versions necessary for your project. These can be easily installed using the provided command.
6. __setup.py__: This file is typically used for package installation and distribution if you plan to package your project for distribution.
7. __source__: The source directory holds the core components of your project:
+ __components__: Contains various Python modules like __data_loading.py__, __keras_tuning.py__, __model_training.py__, and __transformation.py__, responsible for different aspects of the machine learning pipeline.
+ __exception.py__: This file may contain custom exception classes for handling errors or exceptions specific to your project.
+ __loggers.py__: Contains code related to logging and log handling.
+ __pipelines.py__: This directory likely includes a __testing_pipeline.py__ module that defines a pipeline for testing your models.
+ __utils.py__: Contains utility functions or helper code utilized throughout the project.
8. __static__: This directory holds static files such as images (__iris-setosa.png__, __iris-versicolor.jpeg__, __iris-virginica.png__, and __main.jpeg__), which are potentially used in the user interface or documentation.
   
## 4. :gear: __Run Locally__

Clone the project

```bash
git clone https://github.com/software-babooi/ideal-potato-oibsip/tree/main/t1-iris-classifier
```

Go to the project directory

```bash
cd my-project
```

Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

Initiate training

```bash
python3 source/components/data_loading.py
```
Initiating user interface

```bash
python3 -m streamlit run app.py
```
## 5. :ship: __Deployment via Pulling the Docker Image__

Before proceeding with the deployment using Docker, it is essential to ensure that port 9999 is available on your system. The Docker image has been built using this specified port number. If you wish to change the port number, follow the deployment process using Docker as outlined below. This will help avoid conflicts and ensure seamless communication between the Docker container and your system.

+ __Pulling the image from my registry__:

<p>&nbsp  &nbsp  &nbsp  &nbsp  &nbspBefore proceeding with pulling the image, kindly take note that the size of the image is approximately 3.99GB.</p>

```bash
docker pull guna1/oibsip-t1-iris-classifier:latest
```



+ __Running the image__:

```bash
docker run -p 9999:8501 guna1/oibsip-t1-iris-classifier:latest
```

## 6. :ship: __Deployment using Docker__

To implement the deployment of this project in Docker and change the port number, please follow the instructions below:

1. __Check Available Ports__:
First, check the available ports on your system to find one that is not in use.

2. __Replace Port Number in Dockerfile__:
In the Dockerfile, locate the line with "EXPOSE 9999" (where 9999 is the default port). Replace this number with the desired port number from step 1.

<p>&nbsp  &nbsp  &nbsp  &nbsp  &nbspBy following these instructions, you can set the Docker container to use your desired system port, ensuring smooth deployment and communication without conflicts.
</p>

:rotating_light:  The following Docker deployment consumes approximately 3.99GB of the Docker disk image.

+ __Build the docker image__:
  
  <p>&nbsp  &nbsp  &nbsp  &nbsp  &nbsp  Execute the following command to build the Docker image, replacing tagname with an appropriate tag:
  </p>   
```bash
docker build -t tagname .
```
+ __Run the Docker Container__:

    <p>&nbsp  &nbsp  &nbsp  &nbsp  &nbsp  Start the Docker container using the built image with the following command:
    Starting the image
    </p>

```bash
docker run -p (your_desired_port):8501 tagname:latest
```

+ __Pushing the Image to a Docker Registry__:

    <p>&nbsp  &nbsp  &nbsp  &nbsp  &nbsp  To avoid redeploying the image each time, you can push it to your Docker registry. Before pushing, make sure you are logged in to your Docker account using:
    </p>
```bash
docker login
```

<p>Next, push the image to your registry with:</p>

```bash
docker push account_name/image_name:latest
```
This process allows you to store and retrieve the image from the registry during future deployments, making it more efficient and accessible.
## 7. 📽 __Demo__
<div align='center'>
<img src='https://github.com/software-babooi/ideal-potato-oibsip/assets/110555361/4aa53103-d785-4125-a435-a309d967c0a8' width=250 height=250>
</div>




