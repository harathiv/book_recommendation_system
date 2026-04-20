import pickle
import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz

@st.cache_resource
def load_artifacts():
    popular_df = pd.read_pickle("artifacts/popular_df.pkl")
    book_df = pd.read_pickle("artifacts/book_df.pkl")

    vectorizer = joblib.load("artifacts/vectorizer.pkl")
    book_nn = joblib.load("artifacts/book_nn.pkl")
    book_matrix = load_npz("artifacts/book_matrix.npz")

    user_nn = joblib.load("artifacts/user_nn.pkl")
    user_item_sparse = load_npz("artifacts/user_item_sparse.npz")

    with open("artifacts/user_ids.pkl", "rb") as f:
        user_ids = pickle.load(f)

    with open("artifacts/book_titles_cf.pkl", "rb") as f:
        book_titles_cf = pickle.load(f)

    with open("artifacts/books_meta.pkl", "rb") as f:
        books_meta = pickle.load(f)

    book_titles = sorted(book_df["title"].dropna().unique().tolist())

    return (
        popular_df,
        book_df,
        vectorizer,
        book_nn,
        book_matrix,
        user_nn,
        user_item_sparse,
        user_ids,
        book_titles_cf,
        books_meta,
        book_titles,
    )


(
    popular_df,
    book_df,
    vectorizer,
    book_nn,
    book_matrix,
    user_nn,
    user_item_sparse,
    user_ids,
    book_titles_cf,
    books_meta,
    book_titles,
) = load_artifacts()


def get_book_info(title):
    info = books_meta[books_meta["title"] == title]
    author = info.iloc[0]["author"] if not info.empty else "Unknown"

    rating_row = popular_df[popular_df["title"] == title]
    rating = round(rating_row.iloc[0]["avg_rating"], 2) if not rating_row.empty else "N/A"

    return {
        "title": title,
        "author": author,
        "rating": rating
    }


def recommend(book_name):
    if not book_name:
        return []

    query = book_name.strip().lower()

    matches = book_df[book_df["title"].str.lower() == query]

    if matches.empty:
        matches = book_df[book_df["title"].str.lower().str.contains(query, na=False)]

    if matches.empty:
        return []

    idx = matches.index[0]

    distances, indices = book_nn.kneighbors(book_matrix[idx], n_neighbors=6)

    recommendations = []
    for i in indices[0]:
        title = book_df.iloc[i]["title"]
        if title != matches.iloc[0]["title"]:
            recommendations.append(title)

    return recommendations[:5]


def recommend_cf(user_id):
    if user_id not in user_ids:
        return list(popular_df["title"].head(5).values)

    row_idx = user_ids.index(user_id)

    distances, indices = user_nn.kneighbors(user_item_sparse[row_idx], n_neighbors=6)

    recommended_books = []

    for neighbor_idx in indices[0][1:]:
        neighbor_row = user_item_sparse[neighbor_idx]
        cols = neighbor_row.indices
        vals = neighbor_row.data

        for col, val in zip(cols, vals):
            if val >= 3:
                recommended_books.append(book_titles_cf[col])

    # remove duplicates, keep order
    seen = set()
    final = []
    for book in recommended_books:
        if book not in seen:
            seen.add(book)
            final.append(book)

    return final[:5]


def hybrid_recommend(user_id=None, book_name=None):
    if user_id and book_name:
        content_books = recommend(book_name)
        cf_books = recommend_cf(user_id)

        books_list = [book_name]
        for b in content_books:
            if b != book_name and b not in books_list:
                books_list.append(b)

        for b in cf_books:
            if b != book_name and b not in books_list:
                books_list.append(b)

    elif user_id:
        books_list = recommend_cf(user_id)

    elif book_name:
        content_books = recommend(book_name)

        books_list = [book_name]
        for b in content_books:
            if b != book_name and b not in books_list:
                books_list.append(b)

    else:
        books_list = list(popular_df["title"].head(6).values)

    final_output = [get_book_info(book) for book in books_list[:6]]
    return final_output