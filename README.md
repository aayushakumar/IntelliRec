
# IntelliRec :  AI-Powered Product Recommendation System

This is an advanced AI-powered product recommendation system designed to deliver personalized product suggestions based on user queries and preferences. Leveraging cutting-edge machine learning libraries like PyTorch, Transformers, and FAISS, this system combines semantic understanding, fast similarity search, and continuous learning to provide highly relevant recommendations.

## Features

- **Semantic Embeddings**: Generates meaningful representations of product descriptions using a pretrained transformer model (`sentence-transformers/all-MiniLM-L6-v2` by default).
- **Personalization Neural Network**: A deep learning model that tailors recommendations by learning user preferences from feedback.
- **Fast Similarity Search**: Utilizes FAISS to efficiently retrieve similar products based on semantic embeddings.
- **Continuous Learning**: Incorporates user feedback to improve recommendation quality over time.
- **Flexible Design**: Supports customization of embedding models and personalization network architecture.

## Requirements

To run this project, you’ll need the following dependencies:

- **Python 3.7+**
- **PyTorch**
- **Transformers** (Hugging Face)
- **FAISS** (CPU or GPU version, depending on your setup)
- **NumPy**
- **Pandas**

Install the required packages using:

```bash
pip install -r requirements.txt
```

*Note*: Ensure you have a `requirements.txt` file with the exact versions of these libraries. Alternatively, you can manually install them with `pip install torch transformers faiss-cpu numpy pandas`.

## Usage

Here’s how to set up and use the recommendation system:

### 1. Initialize the Recommender

```python
from recommender import AIProductRecommender

recommender = AIProductRecommender()
```

By default, it uses the `sentence-transformers/all-MiniLM-L6-v2` model and automatically selects CUDA if available, otherwise falls back to CPU.

### 2. Build the Product Index

Prepare a list of `EnhancedProduct` objects and build the FAISS index:

```python
from recommender import EnhancedProduct, ProductFeature, ProductCategory

products = [
    EnhancedProduct(
        title="Dell XPS 15 Laptop",
        price=1999.99,
        platform="Dell",
        features=[
            ProductFeature(name="processor", value="i7-11800H"),
            ProductFeature(name="ram", value="16GB"),
            ProductFeature(name="storage", value="1TB SSD")
        ],
        category=ProductCategory.COMPUTING
    ),
    # Add more products as needed
]

recommender.build_product_index(products)
```

### 3. Generate Recommendations

Get personalized product recommendations for a user query:

```python
recommendations = recommender.recommend_products(
    query="powerful laptop for work",
    top_k=5,
    user_id="user123"
)
```

This returns a list of `EnhancedProduct` objects ranked by relevance and personalized scores (if `user_id` is provided).

### 4. Log User Feedback

Improve recommendations by logging user interactions:

```python
recommender.log_user_feedback(
    user_id="user123",
    product=recommendations[0],
    rating=4.5,
    interaction_type="purchase"
)
```

The system periodically retrains the personalization model every 100 feedback entries.

## Architecture Overview

The recommendation system works as follows:

1. **Embedding Generation**: Product descriptions (titles and features) are converted into semantic embeddings using a pretrained transformer model.
2. **Indexing**: Embeddings are stored in a FAISS index for fast similarity search.
3. **Query Processing**: A user query is embedded and matched against the product index to find similar items.
4. **Personalization**: A neural network adjusts recommendation scores based on user preferences learned from feedback.

This modular design ensures both scalability and adaptability.

## Customization

- **Embedding Model**: Change the transformer model by passing a different `embedding_model_name` to `AIProductRecommender` (e.g., `"distilbert-base-uncased"`).
- **Personalization Network**: Modify the architecture in the `_build_personalization_network` method (e.g., adjust `hidden_dims` or add layers).

## Example

Here’s a full example:

```python
from recommender import AIProductRecommender, EnhancedProduct, ProductFeature, ProductCategory

# Initialize
recommender = AIProductRecommender()

# Define products
products = [
    EnhancedProduct(
        title="Dell XPS 15 Laptop",
        price=1999.99,
        platform="Dell",
        features=[
            ProductFeature(name="processor", value="i7-11800H"),
            ProductFeature(name="ram", value="16GB"),
            ProductFeature(name="storage", value="1TB SSD")
        ],
        category=ProductCategory.COMPUTING
    )
]

# Build index
recommender.build_product_index(products)

# Get recommendations
recommendations = recommender.recommend_products("powerful laptop for work", top_k=3)

# Log feedback
recommender.log_user_feedback("user123", recommendations[0], 4.5, "view")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
