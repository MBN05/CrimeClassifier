import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocessing: Handle missing values
train_df['crimeaditionalinfo'] = train_df['crimeaditionalinfo'].fillna('')
test_df['crimeaditionalinfo'] = test_df['crimeaditionalinfo'].fillna('')

# Handle missing values in target variables ('sub_category')
train_df.dropna(subset=['sub_category'], inplace=True)
test_df.dropna(subset=['sub_category'], inplace=True)

# Separate the target variable and features for both training and testing
X_train = train_df['crimeaditionalinfo']
y_train_category = train_df['category']
y_train_sub_category = train_df['sub_category']

X_test = test_df['crimeaditionalinfo']
y_test_category = test_df['category']
y_test_sub_category = test_df['sub_category']

# Create a pipeline to vectorize the text and train a Naive Bayes classifier for Category
category_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
category_pipeline.fit(X_train, y_train_category)

# Create a pipeline to vectorize the text and train a Naive Bayes classifier for Sub-Category
sub_category_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
sub_category_pipeline.fit(X_train, y_train_sub_category)

# Make predictions on the test set
category_predictions = category_pipeline.predict(X_test)
sub_category_predictions = sub_category_pipeline.predict(X_test)

# Print classification reports with zero_division set to 1 to suppress warnings
print("Category Classification Report:")
print(classification_report(y_test_category, category_predictions, zero_division=1))

print("\nSub-Category Classification Report:")
print(classification_report(y_test_sub_category, sub_category_predictions, zero_division=1))

# Compute and print model accuracy
category_accuracy = category_pipeline.score(X_test, y_test_category)
sub_category_accuracy = sub_category_pipeline.score(X_test, y_test_sub_category)

print(f"\nCategory Model Accuracy: {category_accuracy}")
print(f"Sub-Category Model Accuracy: {sub_category_accuracy}")

# Sample prediction
sample_text = ["The above fraudster is continuously messaging me and Asking me to pay him money or he will send fake cropped nude photos of me to my other contacts through WhatsApp"]
category_pred = category_pipeline.predict(sample_text)
sub_category_pred = sub_category_pipeline.predict(sample_text)

print("\nSample Prediction:")
print(f"Category: {category_pred[0]}")
print(f"Sub-Category: {sub_category_pred[0]}")
