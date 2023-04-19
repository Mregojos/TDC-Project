# main.py
#--------------Import Libraries------------------------------#
import streamlit as st
import streamlit_option_menu as menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib


#--------------Page Config-----------------------------------#
st.set_page_config(page_title="Testosterone Deficiency Classificatio")
st.header("Testosterone Deficiency Prediction")

#--------------Menu------------------------------------------#
menu_style = {'background-color': 'red'}
selected_option = menu.option_menu("", options=["Prediction", "Model Evaluation", "Data Visualization"], orientation='horizontal', icons=["", "bi-bar-chart", "bi-graph-up"])

#--------------Variables------------------------------------#
models = {"Logistic Regression": "models/logistic_regression_model.pkl",
          "Decision Tree": "models/decision_tree_model.pkl", 
          "Random Forest": "models/random_forest_model.pkl",
          "Gradient Boosting": "models/gradient_boosting_model.pkl"}

#--------------Model Prediction-----------------------------#
if selected_option == "Prediction":
    # Load the Model and the Scaler files
    model = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])
    loaded_model = joblib.load(models[model])
    loaded_scaler = joblib.load("models/scaler.pkl")

    def predict(data):
        data_scaled = loaded_scaler.transform(data)
        prediction = loaded_model.predict(data_scaled)
        return prediction

    st.subheader("Input Patient Data")

    # Input fields
    age = st.number_input("Age", min_value=45, max_value=85, step=1)
    dm = st.radio("Diabetes", options=["No", "Yes"])
    tg = st.number_input("Triglycerides (TG)", min_value=0, max_value=100, step=1)
    ht = st.radio("Hypertension", options=["No", "Yes"])
    hdl = st.number_input("HDL", min_value=0, max_value=200, step=1)
    ac = st.number_input("Waist Circumference ", min_value=0, max_value=250, step=1)

    predict_button = st.button("Predict")

    choice = {"No": 0,
              "Yes": 1}

    if predict_button:
        input_data = pd.DataFrame({
            "Age": [age],
            "DM": [choice[dm]],
            "TG": [tg],
            "HT": [choice[ht]],
            "HDL": [hdl],
            "AC": [ac]
        })
        prediction = predict(input_data)

        if prediction[0] == 0:
            st.success("The patient is predicted not to have Testosterone Deficiency.")
        else:
            st.warning("The patient is predicted to have Testosterone Deficiency.")


#--------------Model Evaluation-----------------------------#
if selected_option == "Model Evaluation":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Read csv file
    data = pd.read_csv("data/data.csv")
    
    # Split the data into training and testing sets and apply necessary scaling
    # Separate fetures and target
    X = data.drop("T", axis=1)
    y = data["T"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build, train, and evaluate the model
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

    # Create classifiers
    classifiers = {
         "Logistic Regression": LogisticRegression(random_state=42),
         "Decision Tree": DecisionTreeClassifier(random_state=42),
         "Random Forest": RandomForestClassifier(random_state=42),
         "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    

    # Function to train and evaluate a classifier
    def evaluate_classifier(name, clf, X_train, y_train, X_test, y_test):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        st.subheader(f"{name}: ")
        st.write(f"   Accuracy: {acuracy:.2f}")
        st.write(f"   Precision: {precision:.2f}")
        st.write(f"   Recall: {recall:.2f}")
        st.write(f"   F1-score: {f1:.2f}")
        st.write(f"   Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.write(f"   Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
     
    classifier = st.selectbox("Select Classifier", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])
    evaluate_button = st.button("Evaluate")
    if evaluate_button:
        evaluate_classifier(classifier, classifiers[classifier], X_train, y_train, X_test, y_test)
        

#--------------Data Visualization-----------------------------#
if selected_option == "Data Visualization":
    # Read csv file
    data = pd.read_csv("data/data.csv")
    
    visualization = st.selectbox("Select Visualization", ["Plot histogram for each feature", "Plot box plots for each feature", "Plot scatter plots between each pair of features", "Compute the correlation matrix", "Plot the histogram of the target variable (Testosterone)", "About the Data"])
        
    if visualization == "Plot histogram for each feature":
        # Plot histogram for each feature
        data.hist(figsize=(12, 10))
        st.pyplot(plt)
        
    elif visualization == "Plot box plots for each feature":
        # Plot box plots for each features
        plt.figure(figsize=(12, 10))
        for i, col in enumerate(data.columns, 1):
            plt.subplot(3, 3, i)
            data.boxplot(column=col)
            plt.title(col)
        st.pyplot(plt)
     
    elif visualization == "Plot scatter plots between each pair of features":
        # Plot scatter plots between each pair of features
        sns.pairplot(data, diag_kind='kde')
        st.pyplot(plt)

    elif visualization == "Compute the correlation matrix":
        # Compute the correlation matrix
        correlation_matrix = data.corr()

        # Plot the correlation matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt)
    
    elif visualization == "Plot the histogram of the target variable (Testosterone)":
        # Analyze the target variable by creating a histogram
        # Plot the histogram of the target variable (Testosterone)
        plt.figure(figsize=(8, 6))
        data["T"].hist()
        plt.xlabel("Testosterone")
        plt.ylabel("Frequency")
        plt.title("Testosterone Distribution")
        st.pyplot(plt)
        
    else:
        st.write("""
        Features
        * Age: (45-85 years)
        * Triglycerides (TG): (mg/dl)
        * Waist Circumference (WC / AC): (cm)
        * HDL: (mg/dl)
        * Hypertention (HT): (1/0)
        * Diabetes (DM): (1/0)

        Target:
        * T: Testosterone (Medical Literature suggests a defeciency when T < 300ng/dl)

        [Project GitHub](https://github.com/mregojos)
        """)
