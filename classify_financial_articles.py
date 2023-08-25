import pickle
import pandas as pd

model = pickle.load(open(''))
tfidf_vectorizer = pickle.load(open(''))
label_encoder = pickle.load(open(''))

def process(inPath, outPath): # CSV file 'inPath'
    # Read input file
    input_df = pd.read_csv(inPath)
    # Vectorize the data
    features = tfidf_vectorizer.transform(input_df['body'])
    # Predict the classes
    predictions = model.predict(features)
    # Convert output labels to categories
    input_df['category'] = label_encoder.inverse_transform(predictions)
    # Save result to csv
    output_df = input_df['id', 'category']
    output_df.to_csv(outPath, index = False)