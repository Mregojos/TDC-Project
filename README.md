# Testosterone Deficiency Prediction

## Objective
* Build a classification model to classify Testosterone Deficiency
* Explore and visualize the data 
* Build and deploy a web app to classify Testosterone Deficiency

## Tech Stack
* Python, Jupyterlab, Pandas, NumPy, Matplotlib, Streamlit, Docker

## Data
* Testosterone Deficiency Classification

## Tasks
```sh
# Clone the repository
git clone https://github.com/Mregojos/TDC-Project
cd TDC-Project

# Build and run the web app
docker build -t tdp-web-app .
docker run -d --name tdp-web-app -p 8501:8501 tdp-web-app
```


```sh
# To run the web app (with volume) and jupyterlab
docker build -t tdp-web-app .
docker run -d --name tdp-web-app -p 8501:8501 -v $(pwd):/app tdp-web-app

# Jupyterlab
cd jupyterlab-docker
docker build -t jupyterlab .
cd ..
docker run --name jupyterlab -p 8888:8888 -v $(pwd):/app jupyterlab


# Delete containers
docker rm -f tdp-web-app
docker rm -f jupyterlab
```

![TDP](https://github.com/Mregojos/TDC-Project/blob/main/images/TDP.png)

## Reference
[Testosterone Deficiency Classification Data](https://github.com/osmarluiz/testosterone-deficiency-dataset)

[(Science Direct) Prediction of secondary testosterone deficiency using Machine Learning: A comparative analysis of 
ensemble and base classifiers, probability calibration, and sampling strategies in a slightly imbalanced dataset.](https://sciencedirect.com/science/article/pii/S235291821000289)
