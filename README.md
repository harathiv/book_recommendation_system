# book_recommendation_system
A hybrid book recommendation system built using machine learning techniques including popularity-based, content-based, collaborative filtering, and hybrid models. The system features a user-friendly Streamlit interface with login/registration, enabling users to explore personalized and similarity-based book recommendations.

## Recommendation data flow

The CSV files provide the historical training data. `prepare_artifacts.py` cleans that data, creates the historical user-book rating matrix, trains the nearest-neighbor models, and saves the reusable files in `artifacts/`.

The Streamlit application uses SQLite for live application data:

- `users.db` stores registered users and their saved book ratings.
- A user's rating is stored in the `user_ratings` table when they use the **Save Rating** control.
- Personalized and hybrid recommendations combine the historical artifact matrix with live SQLite rating rows.
- Users with no ratings receive popularity-based recommendations until they have enough interactions for personalized matching.

The live matrix is rebuilt from SQLite when a recommendation is requested. This keeps the prototype simple and ensures that ratings from multiple application users can influence collaborative recommendations without modifying the original CSV files.

## Running the application

```bash
streamlit run app.py
```

To rebuild the historical artifacts after changing the source CSV data:

```bash
python prepare_artifacts.py
```
