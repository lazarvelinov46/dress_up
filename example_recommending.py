import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Simulate a database for clothes with picture, type, name, link, price, color, and material
class ClothesDatabase:
    def __init__(self):
        # Load your dataset here
        self.data = pd.read_csv('clothing_fit_and_style_dataset.csv')  # Example dataset

    def add_item(self, item):
        # Add a new item to the database
        self.data = self.data.append(item, ignore_index=True)

    def get_filtered_items(self, item_type=None, price_range=None):
        filtered_items = self.data.copy()
        if item_type:
            filtered_items = filtered_items[filtered_items['type'] == item_type]
        if price_range:
            filtered_items = filtered_items[
                (filtered_items['price'] >= price_range[0]) & 
                (filtered_items['price'] <= price_range[1])
            ]
        return filtered_items

# Define a function to recommend clothes based on user's wardrobe
def recommend_clothes(user_wardrobe, clothes_db, top_n=10):
    dataset_descriptions = clothes_db.data['description'].fillna('')
    user_descriptions = user_wardrobe['description'].fillna('')
    
    # Combine descriptions for TF-IDF vectorization
    all_descriptions = dataset_descriptions.tolist() + user_descriptions.tolist()
    
    # Vectorize descriptions using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)
    
    # Split the TF-IDF matrix back into dataset and user matrices
    dataset_tfidf_matrix = tfidf_matrix[:len(dataset_descriptions)]
    user_tfidf_matrix = tfidf_matrix[len(dataset_descriptions):]
    
    # Train an SVM classifier
    X_train = user_tfidf_matrix
    y_train = [1] * len(user_wardrobe)  # Label all items in the user's wardrobe as relevant
    model = make_pipeline(StandardScaler(with_mean=False), SVC(kernel='linear', probability=True))
    model.fit(X_train, y_train)
    
    # Predict on the dataset
    predictions = model.decision_function(dataset_tfidf_matrix)
    clothes_db.data['predicted_score'] = predictions
    
    # Sort by predicted score in descending order and recommend top N clothes
    recommended_clothes = clothes_db.data.sort_values(by='predicted_score', ascending=False).head(top_n)
    return recommended_clothes[['description', 'predicted_score']]

# Define a function to pair items based on selected item
def pair_items(selected_item, clothes_db):
    dataset_descriptions = clothes_db.data['description'].fillna('')
    selected_item_desc = selected_item['description']
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset_descriptions.tolist() + [selected_item_desc])
    
    dataset_tfidf_matrix = tfidf_matrix[:-1]
    selected_item_tfidf_matrix = tfidf_matrix[-1]
    
    similarity_matrix = cosine_similarity(dataset_tfidf_matrix, selected_item_tfidf_matrix.reshape(1, -1))
    clothes_db.data['similarity'] = similarity_matrix
    paired_items = clothes_db.data.sort_values(by='similarity', ascending=False)
    
    return paired_items[['description', 'similarity']]

# Example usage
def main():
    # Initialize the clothes database
    clothes_db = ClothesDatabase()
    
    # Example user wardrobe
    user_wardrobe = pd.DataFrame({
        'description': [
            'Red cotton dress with floral patterns',
            'Blue denim jeans with ripped knees'
        ]
    })
    
    # Get recommendations based on user wardrobe
    recommended = recommend_clothes(user_wardrobe, clothes_db)
    print("Recommended Clothes:")
    print(recommended)
    
    # Example selected item to pair with others
    selected_item = {'description': 'Red cotton dress with floral patterns'}
    paired = pair_items(selected_item, clothes_db)
    print("Paired Clothes:")
    print(paired)

if __name__ == "__main__":
    main()
