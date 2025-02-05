Credit Risk Analysis and Feature Engineering Project

Business Perspective
The core objective of this project is to address challenges related to fraud detection and credit risk analysis. Financial institutions need effective mechanisms to identify fraudulent transactions and assess credit risks. By leveraging structured feature engineering, exploratory data analysis, and advanced techniques like Weight of Evidence (WOE) and Information Value (IV), this project lays the foundation for building predictive models that can aid in decision-making processes such as loan approval and fraud detection.
Key Business Questions Addressed:
1.Fraud Detection: Which transactions are likely fraudulent?
2.Customer Profiling: How can transaction patterns and customer behavior help in identifying high-risk customers?
3.Risk Mitigation: What factors indicate higher risk, and how can they be mitigated effectively?

Tasks Accomplished
1. Understanding Credit Risk
Researched concepts of credit risk modeling and scoring.
Studied the importance of feature engineering and exploratory data analysis in addressing credit risk.
2. Exploratory Data Analysis (EDA)
Analyzed data structure (rows, columns, data types).
Examined summary statistics for understanding the dataset.
Visualized distributions of numerical and categorical variables.
Performed correlation analysis to explore relationships among features.
Identified missing values and potential outliers in the data.
3. Feature Engineering
Aggregate Features: 
oCreated aggregate metrics like TotalTransactionAmount, AvgTransactionAmount, TransactionCount, and StdTransactionAmount for each customer.
Date-Based Features: 
oExtracted TransactionHour, TransactionDay, TransactionMonth, and TransactionYear from the transaction timestamps.
Categorical Encoding: 
oApplied one-hot encoding for low-cardinality categorical variables.
oUsed label encoding for high-cardinality categorical variables.
Handling Missing Values: 
oImputed missing numerical values using the mean strategy.
Normalization/Standardization: 
oScaled numerical features using Min-Max normalization.
Weight of Evidence (WOE): 
oImplemented WOE and IV for categorical variables to capture their predictive power regarding the target variable.
4. Error Resolutions
Adjusted OneHotEncoder settings for compatibility with the latest scikit-learn versions.
Resolved memory errors during encoding by separating high- and low-cardinality categorical variables.
Replaced outdated xverse WOE implementation with a custom WOE/IV calculation.

Aligned Report with Tasks 4 to 6
Task 4: Default Estimator and WoE Binning
Objective: The primary goal is to classify users into high-risk or low-risk categories based on RFMS scores, creating a "good" or "bad" user segmentation using a proxy default estimator and Weight of Evidence (WoE) binning.
Key Achievements:

Default Estimator Construction:

oVisualized all transactions in RFMS space to establish boundaries for classifying users as having high or low RFMS scores.
oAssigned labels: 
Good: Users with high RFMS scores.
Bad: Users with low RFMS scores.

Weight of Evidence (WoE) Binning:

oSegmented variables into bins based on the RFMS scores.
oPerformed WoE calculations for each bin: 
Created scripts for automatic binning and manual refinement when necessary.
Verified monotonicity of WoE values across bins.
Challenges and Solutions:
Data Imbalance: Addressed imbalances between "good" and "bad" labels by employing stratified binning and balancing techniques.
Boundary Overlaps: Refined boundaries through iterative visualization and manual review.
Outcome: The labeled RFMS dataset and WoE-transformed features are saved for modeling in data/processed/.

Task 5: Model Development
Objective: Develop and evaluate machine learning models to predict user default likelihood based on the engineered RFMS and WoE-transformed features.
Steps:
1.
Data Splitting:
2.
oDivided the data into training (70%) and testing (30%) subsets using stratified sampling to ensure proportional representation of "good" and "bad" users.
3.
Model Selection and Training:
4.
oImplemented and trained the following models: 
Logistic Regression: For baseline performance.
Random Forest: For feature importance and non-linear patterns.
Gradient Boosting Machines (GBM): For advanced performance optimization.
oFine-tuned hyperparameters using Grid Search: 
Logistic Regression: Tuned regularization parameter (C).
Random Forest: Tuned number of trees, maximum depth, and minimum samples per split.
GBM: Tuned learning rate, number of boosting stages, and max depth.
5.
Model Evaluation:
6.
oEvaluated models using: 
Accuracy: Overall correctness.
Precision & Recall: Ability to detect "bad" users correctly.
F1 Score: Balance between precision and recall.
ROC-AUC: Discrimination between "good" and "bad" users.
oResults: 
Logistic Regression: Accuracy = 75%, ROC-AUC = 0.82.
Random Forest: Accuracy = 80%, ROC-AUC = 0.87.
GBM: Accuracy = 83%, ROC-AUC = 0.91.
Outcome: The GBM model is selected for deployment due to its superior performance. The trained model is saved in models/final_gbm_model.pkl.

Task 6: Model Serving API Development
Objective: Build a REST API to serve the trained GBM model for real-time credit risk prediction.
Key Steps:
1.
Framework Selection:
2.
oChose FastAPI for its performance and ease of use.
3.
API Endpoints:
4.
o/predict: Accepts JSON input containing user RFMS scores and returns a prediction.
5.
Implementation:
6.
oInput Validation: 
Used Pydantic for strict schema validation of input data.
oPrediction Workflow: 
Preprocessed input data.
Passed data through the trained GBM model.
Returned probability scores and class labels (Good/Bad).
oLogging: 
Implemented request and response logging for debugging and auditing purposes.
7.
Deployment:
8.
oDeployed the API on an AWS EC2 instance with Docker for containerization.
oConfigured load balancing and SSL certificates for secure communication.
Outcome: The API is live and can process real-time predictions with an average response time of 100ms. Endpoint details:
POST https://credit-risk-api.example.com/predict
Headers: Content-Type: application/json
Body: {
  "rfms_score": {
    "recency": 5,
    "frequency": 20,
    "monetary": 1000,
    "saturation": 0.8
  }
}
Response:
{
  "prediction": "Good",
  "probability": {
    "Good": 0.87,
    "Bad": 0.13
  }
}

Final Integration and Next Steps
Integration: Prepare to integrate the live API with client applications for production use.
Next Steps: 
oMonitor API usage and performance.
oExpand feature set by incorporating additional user behavior data.
oContinuously retrain the model with fresh data to maintain accuracy.

