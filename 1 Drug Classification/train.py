import pandas as pd
import skops.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Load the dataset
drug_df = pd.read_csv("data/drug.csv")
drug_df = drug_df.sample(frac=1)
drug_df.head(3)


# Define the features and label columns
X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# Create Pipeline
cat_col = [1,2,3]
num_col = [0,4]

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)
pipe.fit(X_train, y_train)

# Model evaluation
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))


# Create Confusion Matrix and save the figure in the Results folder.
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)

# Create the metrics file and save it in the Results folder.
with open("results/metrics.txt", "w") as outfile:
    # Metrics:
    accuracy_rounded = round(accuracy, 2)
    f1_rounded = round(f1, 2)
    outfile.write(f"\nAccuracy = {accuracy_rounded}, F1 Score = {f1_rounded}.")
    print("Metrics saved in Results folder.")
    # Confusion Matrix:
    outfile.write(f"\nConfusion Matrix:\n{cm}")

# Save the model and pipeline with sio. This will help us save both the scikit-learn pipeline and model.
model_pipeline_path = "model/drug_pipeline.skops"
sio.dump(pipe, model_pipeline_path)

# Load the model and pipeline
sio.load(model_pipeline_path, trusted=True)