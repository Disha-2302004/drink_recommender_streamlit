
import streamlit as st
import pandas as pd
import json, os, hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Config & paths
# -------------------------
DATA_FILE = "cleaned_carbonated_drinks.csv"
USERS_FILE = "users.json"
FEEDBACK_FILE = "feedback.json"

# -------------------------
# Helpers
# -------------------------
def load_json(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return {}

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

def hash_password(password: str):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# -------------------------
# Build simple TF-IDF recommender
# -------------------------
@st.cache_resource
def build_vectorizer(df):
    # combine useful textual features
    df["features"] = df.get("Flavor", "").fillna("") + " " + df.get("Ingredients", "").fillna("") + " " + df.get("Category","").fillna("")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    X = vectorizer.fit_transform(df["features"])
    return vectorizer, X

def recommend(df, vectorizer, X, query, top_k=6, user_history=None):
    # incorporate user history by appending it to the query (simple personalization)
    if user_history:
        history_text = " ".join(user_history[-5:])
        final_query = query + " " + history_text
    else:
        final_query = query
    qv = vectorizer.transform([final_query])
    sims = cosine_similarity(qv, X).flatten()
    idx = sims.argsort()[-top_k:][::-1]
    return df.iloc[idx].copy(), sims[idx]

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="Carbonated Drink Recommender", layout="wide")

# ensure JSON files exist
for f, default in [(USERS_FILE, {}), (FEEDBACK_FILE, {})]:
    if not os.path.exists(f):
        save_json(f, default)

# Load data
if not os.path.exists(DATA_FILE):
    st.error(f"Dataset file '{DATA_FILE}' not found in app folder. Please add it and reload.")
    st.stop()

df = load_data(DATA_FILE)
vectorizer, X = build_vectorizer(df)

# Sidebar navigation
menu = ["Login/Signup", "Recommendations", "Feedback", "About"]
choice = st.sidebar.selectbox("Navigation", menu)

# -------------------------
# Auth functions
# -------------------------
users = load_json(USERS_FILE)

def signup(username, password):
    users = load_json(USERS_FILE)
    if username in users:
        return False, "Username exists"
    users[username] = {"password": hash_password(password), "history": [], "favorites": []}
    save_json(USERS_FILE, users)
    return True, "Signup successful"

def login(username, password):
    users = load_json(USERS_FILE)
    if username in users and users[username]["password"] == hash_password(password):
        return True, "Login successful"
    return False, "Invalid credentials"

# -------------------------
# Pages
# -------------------------
if choice == "Login/Signup":
    st.title("🔐 Login / Signup")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Login")
        u_login = st.text_input("Username", key="login_user")
        p_login = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            ok, msg = login(u_login, p_login)
            if ok:
                st.session_state["user"] = u_login
                st.success(msg)
            else:
                st.error(msg)
    with col2:
        st.subheader("Signup")
        u_signup = st.text_input("New username", key="signup_user")
        p_signup = st.text_input("New password", type="password", key="signup_pass")
        if st.button("Signup"):
            ok, msg = signup(u_signup, p_signup)
            if ok:
                st.success(msg + ". Now login from left panel.")
            else:
                st.error(msg)

elif choice == "Recommendations":
    st.title("🥤 Personalized Recommendations")
    if "user" not in st.session_state:
        st.warning("Please login first from the Login/Signup page.")
    else:
        user = st.session_state["user"]
        users = load_json(USERS_FILE)  # reload to get latest
        history = users.get(user, {}).get("history", [])

        st.markdown(f"**Logged in as:** {user}")
        query = st.text_input("Describe what you want (e.g., low sugar, ginger, caffeine-free, post-workout):", key="query_input")

        col1, col2 = st.columns([1,3])
        with col1:
            top_k = st.number_input("Number of results", min_value=1, max_value=12, value=6)
            if st.button("Get Recommendations"):
                if not query.strip():
                    st.warning("Please enter a query.")
                else:
                    results, scores = recommend(df, vectorizer, X, query, top_k=int(top_k), user_history=history)
                    # Save user query to history
                    users = load_json(USERS_FILE)
                    users.setdefault(user, {"password":"", "history":[], "favorites":[]})
                    users[user]["history"].append(query)
                    save_json(USERS_FILE, users)

                    st.subheader("Top recommendations")
                    for i, row in results.iterrows():
                        st.write(f"**{row.get('Drink_Name', 'Unknown')}**")
                        st.write("• Flavor: " + str(row.get("Flavor","")))
                        st.write("• Ingredients: " + str(row.get("Ingredients","")))
                        st.write("• Category: " + str(row.get("Category","")))
                        st.write("---")
                    st.success("Recommendations generated.")

        with col2:
            st.subheader("Your recent searches")
            st.write(history[-10:] if history else "No history yet.")
            st.subheader("Favorites")
            favs = users.get(user, {}).get("favorites", [])
            if favs:
                st.write(favs)
            else:
                st.write("No favorites yet.")

elif choice == "Feedback":
    st.title("📝 Feedback")
    if "user" not in st.session_state:
        st.warning("Please login first.")
    else:
        user = st.session_state["user"]
        text = st.text_area("Write feedback about recommendations or the app:")
        if st.button("Submit Feedback"):
            feedbacks = load_json(FEEDBACK_FILE)
            feedbacks.setdefault(user, [])
            feedbacks[user].append({"text": text})
            save_json(FEEDBACK_FILE, feedbacks)
            st.success("Thanks for the feedback!")

elif choice == "About":
    st.title("About this App")
    st.markdown("""
    - This is a simple Streamlit app that recommends carbonated drinks using a TF-IDF text-similarity model.
    - Login/signup data and feedback are stored in `users.json` and `feedback.json`.
    - To deploy to Streamlit Cloud: push the repo to GitHub and connect the repo in Streamlit Cloud. The main file is `app.py`.
    """)

# Footer
st.sidebar.markdown("---")
if "user" in st.session_state:
    if st.sidebar.button("Logout"):
        st.session_state.pop("user", None)
        st.experimental_rerun()
