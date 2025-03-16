This project analyzes and predicts phone usage patterns among users in India based on various features such as screen time, data usage, and demographic factors. It includes data preprocessing, exploratory data analysis (EDA), machine learning models, and a Streamlit-based interactive web app for real-time predictions.

🚀 Features
	•	Data Cleaning & Preprocessing: Handling missing values, encoding categorical variables, and removing outliers.
	•	Exploratory Data Analysis (EDA): Visualizing distributions, correlations, and trends in phone usage.
	•	Feature Engineering: Applying SMOTE for class balancing, scaling, and feature selection using Recursive Feature Elimination (RFE).
	•	Machine Learning Models: Training Random Forest and XGBoost classifiers with Optuna-based hyperparameter tuning.
	•	Interactive Streamlit App: Allows users to input features and instantly get predictions without unnecessary loading time.
	•	Optimized Performance: Efficient caching ensures that input changes and predictions update instantly without reloading the entire app.

📂 Dataset

The dataset includes user phone usage details, such as:
	•	Screen Time (hrs/day)
	•	Data Usage (GB/month)
	•	Calls Duration (mins/day)
	•	Social Media Time (hrs/day)
	•	Streaming Time (hrs/day)
	•	Gaming Time (hrs/day)
	•	E-commerce Spend (INR/month)
	•	Monthly Recharge Cost (INR)
	•	Demographic features (Gender, Phone Brand, etc.)

The target variable classifies users into High Usage (1) or Low Usage (0) categories based on screen time and data usage.

📊 Exploratory Data Analysis (EDA)

The project includes the following EDA visualizations:
	•	Histogram of screen time distribution
	•	Boxplot for data usage outliers
	•	Scatter plot of screen time vs. data usage
	•	Correlation heatmap for feature relationships
	•	Class distribution bar chart

⚙ Model Training & Optimization
	•	Data Balancing: Used SMOTE to handle class imbalance.

 Features of the Web App
	•	Users can select demographic details and enter numerical features.
	•	Predictions update instantly without reloading.
	•	Displays classification results (High or Low Usage) in real time.

📌 Technologies Used
	•	Python (Pandas, NumPy, Scikit-Learn, XGBoost, Seaborn, Matplotlib)
	•	Machine Learning (Random Forest, XGBoost, Hyperparameter Tuning)
	•	Streamlit (Web-based interactive dashboard)

🎯 Future Improvements
	•	Incorporating deep learning models for better accuracy.
	•	Adding real-time data collection and dynamic analysis.
	•	Deploying the app on AWS EC2 for global accessibility.

🏆 Results
	•	Optimized input handling for instant updates.
	•	High accuracy classification of phone usage patterns.
	•	Interactive web interface for better user experience.
	•	Feature Scaling: Applied StandardScaler for normalization.
	•	Feature Selection: Used Recursive Feature Elimination (RFE) to select the top 10 features.
	•	Hyperparameter Tuning: Used Optuna to optimize model parameters dynamically.
	•	Model Evaluation: Trained Random Forest and XGBoost, achieving high accuracy.
