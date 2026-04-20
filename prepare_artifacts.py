import os
import pickle
import joblib
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

os.makedirs("artifacts", exist_ok=True)

# ---------------- LOAD ----------------
books = pd.read_csv(
    "data/Books.csv",
    sep=";",
    encoding="ISO-8859-1",
    on_bad_lines="skip",
    low_memory=False
)

users = pd.read_csv(
    "data/Users.csv",
    sep=";",
    encoding="ISO-8859-1",
    on_bad_lines="skip",
    low_memory=False
)

ratings = pd.read_csv(
    "data/Ratings.csv",
    sep=";",
    encoding="ISO-8859-1",
    on_bad_lines="skip",
    low_memory=False
)

# ---------------- CLEAN ----------------
books.columns = books.columns.str.strip().str.lower()
users.columns = users.columns.str.strip().str.lower()
ratings.columns = ratings.columns.str.strip().str.lower()

users["user-id"] = pd.to_numeric(users["user-id"], errors="coerce")
ratings["user-id"] = pd.to_numeric(ratings["user-id"], errors="coerce")

users = users.dropna(subset=["user-id"])
ratings = ratings.dropna(subset=["user-id"])

users["user-id"] = users["user-id"].astype(int)
ratings["user-id"] = ratings["user-id"].astype(int)

books = books.dropna(subset=["author", "publisher"])

users["age"] = pd.to_numeric(users["age"], errors="coerce")
users["age"] = users["age"].fillna(users["age"].mean())

# ---------------- MERGE ----------------
ratings_with_books = ratings.merge(books, on="isbn")
final_data = ratings_with_books.merge(users, on="user-id")

# ---------------- FILTER ----------------
# Use stronger filtering so deployment is light
user_counts = final_data["user-id"].value_counts()
active_users = user_counts[user_counts > 30].index
filtered_data = final_data[final_data["user-id"].isin(active_users)]

book_counts = filtered_data["title"].value_counts()
popular_books = book_counts[book_counts >= 15].index
filtered_data = filtered_data[filtered_data["title"].isin(popular_books)]

# Keep only explicit ratings
filtered_data = filtered_data[filtered_data["rating"] != 0]

# ---------------- POPULAR DF ----------------
num_ratings = filtered_data.groupby("title").count()["rating"].reset_index()
num_ratings.rename(columns={"rating": "num_ratings"}, inplace=True)

avg_ratings = filtered_data.groupby("title")["rating"].mean().reset_index()
avg_ratings.rename(columns={"rating": "avg_rating"}, inplace=True)

popular_df = num_ratings.merge(avg_ratings, on="title")
popular_df = popular_df.sort_values(by=["avg_rating", "num_ratings"], ascending=False)

# Add author to popular_df
book_meta = filtered_data[["title", "author"]].drop_duplicates()
popular_df = popular_df.merge(book_meta, on="title", how="left")

# ---------------- CONTENT MODEL ----------------
book_df = filtered_data[["title", "author"]].drop_duplicates().reset_index(drop=True)
book_df["features"] = book_df["title"] + " " + book_df["author"]

vectorizer = TfidfVectorizer(stop_words="english")
book_matrix = vectorizer.fit_transform(book_df["features"])

book_nn = NearestNeighbors(metric="cosine", algorithm="brute")
book_nn.fit(book_matrix)

# ---------------- COLLABORATIVE MODEL ----------------
user_book_table = filtered_data.pivot_table(
    index="user-id",
    columns="title",
    values="rating",
    fill_value=0
)

user_ids = user_book_table.index.tolist()
book_titles_cf = user_book_table.columns.tolist()

user_item_sparse = csr_matrix(user_book_table.values)

user_nn = NearestNeighbors(metric="cosine", algorithm="brute")
user_nn.fit(user_item_sparse)

# ---------------- SAVE ARTIFACTS ----------------
popular_df.to_pickle("artifacts/popular_df.pkl")
book_df.to_pickle("artifacts/book_df.pkl")

joblib.dump(vectorizer, "artifacts/vectorizer.pkl")
joblib.dump(book_nn, "artifacts/book_nn.pkl")
save_npz("artifacts/book_matrix.npz", book_matrix)

joblib.dump(user_nn, "artifacts/user_nn.pkl")
save_npz("artifacts/user_item_sparse.npz", user_item_sparse)

with open("artifacts/user_ids.pkl", "wb") as f:
    pickle.dump(user_ids, f)

with open("artifacts/book_titles_cf.pkl", "wb") as f:
    pickle.dump(book_titles_cf, f)

with open("artifacts/books_meta.pkl", "wb") as f:
    pickle.dump(books[["title", "author"]].drop_duplicates(), f)

print("Artifacts created successfully!")
print("Books in model:", len(book_df))
print("Users in collaborative model:", len(user_ids))