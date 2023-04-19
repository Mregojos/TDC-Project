# Testosterone Deficiency Prediction

## Objective
* Build a classification model to classify Tetosterone Deficiency
* Explore and visualize the data 
* Build and deploy a web app to classify Testosterone Deficiency

## Tech Stack
* Python, Jupyterlab, Pandas, NumPy, Matplotlib, Streamlit, Docker

## Data
* Testosterone Deficiency Classification

## Tasks
```sh
Clone the repository
git clone https://github.com/Mregojos/TDC-Project
cd TDC-Project

# Build and run the web app
docker build -t tdp-web-app .
docker run -d --name tdp-web-app -p 8501:8501 tdp-web-app
```

```sh
# To run streamlit app (with volume and jupyterlab
# streamlit
# cd td-classification
docker build -t streamlit-app .
docker run -d --name streamlit-app  -p 8501:8501 -v $(pwd):/app streamlit-app 

# jupyterlab
cd td-classification
cd jupyterlab-docker
docker build -t jupyterlab .
cd ..
docker run --name jupyterlab -p 8888:8888 -v $(pwd):/app jupyterlab


# delete
docker rm -f streamlit-app
docker rm -f jupyterlab
```

## Reference
[Testosterone Deficiency Classification Data]()
