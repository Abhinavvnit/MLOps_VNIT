import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

## enable autologging
mlflow.sklearn.autolog()

df = pd.read_csv("titanic.csv")

df = df.fillna(0)
df.info()

df.head(3)

#Encode categorical variables as dummy
df["gender_enc"]=df["Sex"].astype('category').cat.codes
df["embark_enc"]=df["Embarked"].astype('category').cat.codes

#Specify dependent and independent variables
X = df[["Pclass","Age","gender_enc","embark_enc","Fare","SibSp","Parch"]]
Y = df["Survived"]

#Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
    }

# step:1 initialise the model class
rf = RandomForestClassifier(random_state=42)
# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the grid search
grid_search.fit(X_train, y_train)
    
# Get the best hyperparameters and the corresponding accuracy
best_params = grid_search.best_params_
best_score = grid_search.best_score_
       
# Report the best hyperparameters and accuracy
print(f"Best Hyperparameters: {best_params}")
print(f"Best Cross-Validated Accuracy: {best_score:.4f}")
    
# Optionally, evaluate the model on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
    
# Report the test accuracy
print(f"Test Accuracy: {test_accuracy:.4f}")
