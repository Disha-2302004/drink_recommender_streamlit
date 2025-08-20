import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# Load dataset and model
# -----------------------------
@st.cache_resource
def load_model_and_index():
    df = pd.read_csv("cleaned_carbonated_drinks.csv")
    df.fillna("", inplace=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    drink_embeddings = model.encode(df["FlavorProfile"].tolist(), show_progress_bar=False)

    dimension = drink_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(drink_embeddings))

    return df, model, index, drink_embeddings


df, model, index, drink_embeddings = load_model_and_index()

# -----------------------------
# Helper Functions
# -----------------------------
def get_query_embedding(user_input):
    query = f"{user_input['flavor']} {user_input['use_case']} {user_input['tags']} {user_input['type']}"
    return model.encode([query])[0]


def get_top_k_recommendations(query_embedding, k=5):
    D, I = index.search(np.array([query_embedding]), k)
    return df.iloc[I[0]]


def get_relevant_items(user_input, df):
    relevant = []
    for _, row in df.iterrows():
        if user_input['flavor'].lower() in row['FlavorProfile'].lower():
            relevant.append(row['ProductName'])
    return relevant


def evaluate_model(user_input, recommendations, k):
    relevant = get_relevant_items(user_input, df)
    retrieved = recommendations['ProductName'].tolist()
    retrieved_k = retrieved[:k]

    y_true = [1 if item in relevant else 0 for item in retrieved_k]
    y_pred = [1] * len(retrieved_k)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    accuracy = sum([1 for item in retrieved_k if item in relevant]) / k * 100 if k > 0 else 0
    return accuracy, precision, recall, f1


# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="Drink Recommender", layout="wide")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Login", "Recommendation", "Feedback"])

# -----------------------------
# Page 1: Login
# -----------------------------
if page == "Login":
    st.title("🔑 Login Page")
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")
    if st.button("Login"):
        if username and password:
            st.success(f"Welcome {username} 🎉 You can now go to Recommendation page.")
        else:
            st.error("Please enter both username and password!")

# -----------------------------
# Page 2: Recommendation
# -----------------------------
elif page == "Recommendation":
    st.title("🥤 Personalized Drink Recommender")

    # User Inputs
    st.subheader("Tell us about your preferences:")
    flavor = st.text_input("Flavor Preference (e.g., Ginger, Lemon, Cola)")
    use_case = st.text_input("Use Case (e.g., Gym, Party, Relaxation)")
    tags = st.text_input("Extra Tags (e.g., Sugar-Free, Caffeine-Free, Low Sugar)")
    drink_type = st.selectbox("Type of Drink", ["Soda", "Juice", "Energy Drink", "Other"])

    if st.button("Get Recommendations"):
        if flavor.strip():
            user_input = {"flavor": flavor, "use_case": use_case, "tags": tags, "type": drink_type}
            query_embedding = get_query_embedding(user_input)
            recommendations = get_top_k_recommendations(query_embedding, k=5)

            st.subheader("✅ Top Recommendations for You")
            st.table(recommendations[["ProductName", "FlavorProfile", "Ingredients"]])

            # Show Evaluation
            accuracy, precision, recall, f1 = evaluate_model(user_input, recommendations, k=5)
            st.subheader("📊 Evaluation Metrics")
            st.write(f"**Accuracy:** {accuracy:.2f}%")
            st.write(f"**Precision:** {precision:.2f}")
            st.write(f"**Recall:** {recall:.2f}")
            st.write(f"**F1 Score:** {f1:.2f}")
        else:
            st.warning("Please enter at least a flavor preference.")

# -----------------------------
# Page 3: Feedback
# -----------------------------
elif page == "Feedback":
    st.title("💬 Feedback Page")
    feedback_text = st.text_area("Please provide your feedback about our recommendations:")
    if st.button("Submit Feedback"):
        if feedback_text.strip():
            st.success("Thank you for your feedback! 🙏")
        else:
            st.warning("Feedback cannot be empty!")
