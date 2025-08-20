# Drink Recommender - Streamlit App

This repository contains a Streamlit app that provides personalized carbonated drink recommendations.

## Files
- `app.py` - The Streamlit application (entry point).
- `cleaned_carbonated_drinks.csv` - Dataset (included).
- `users.json` - Created and used by the app to store user accounts and history.
- `feedback.json` - Created and used by the app to store feedback.
- `requirements.txt` - Python dependencies.

## How to deploy to Streamlit Cloud
1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and connect your GitHub repo.
3. Select `app.py` as the main file. Streamlit Cloud will install dependencies from `requirements.txt`.

## Notes
- Passwords are stored hashed (SHA-256) in `users.json`.
- If you want stronger auth in production, connect Firebase or a proper DB.