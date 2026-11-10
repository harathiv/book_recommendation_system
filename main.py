import pickle

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, load_npz, vstack
from sklearn.neighbors import NearestNeighbors

from auth import get_all_user_ratings


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
    historical_user_nn,
    historical_user_item_sparse,
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
        "rating": rating,
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
    _, indices = book_nn.kneighbors(book_matrix[idx], n_neighbors=6)

    recommendations = []
    for i in indices[0]:
        title = book_df.iloc[i]["title"]
        if title != matches.iloc[0]["title"]:
            recommendations.append(title)

    return recommendations[:5]


def _live_user_matrix():
    """Build live SQLite user rows using the historical artifact column order."""
    title_to_column = {title: index for index, title in enumerate(book_titles_cf)}
    live_rows = {}

    for username, title, rating in get_all_user_ratings():
        column = title_to_column.get(title)
        if column is None:
            continue
        if username not in live_rows:
            live_rows[username] = {}
        live_rows[username][column] = float(rating)

    if not live_rows:
        return None, {}

    rows = []
    usernames = {}
    for row_index, (username, ratings) in enumerate(live_rows.items()):
        row = csr_matrix(
            (
                list(ratings.values()),
                ([0] * len(ratings), list(ratings.keys())),
            ),
            shape=(1, len(book_titles_cf)),
        )
        rows.append(row)
        usernames[username] = row_index

    return vstack(rows, format="csr"), usernames


def recommend_cf(username=None, user_id=None):
    live_matrix, live_user_indexes = _live_user_matrix()
    target_matrix = historical_user_item_sparse
    target_row = None
    live_user_count = 0

    if live_matrix is not None:
        target_matrix = vstack([target_matrix, live_matrix], format="csr")
        live_user_count = live_matrix.shape[0]

    if username in live_user_indexes:
        target_row = len(user_ids) + live_user_indexes[username]
    elif user_id in user_ids:
        target_row = user_ids.index(user_id)

    if target_row is None:
        return list(popular_df["title"].head(5).values)

    model = historical_user_nn
    if live_user_count:
        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(target_matrix)

    neighbor_count = min(6, target_matrix.shape[0])
    _, indices = model.kneighbors(target_matrix[target_row], n_neighbors=neighbor_count)

    rated_columns = set(target_matrix[target_row].indices)
    recommended_books = []

    for neighbor_idx in indices[0]:
        if neighbor_idx == target_row:
            continue

        neighbor_row = target_matrix[neighbor_idx]
        for column, rating in zip(neighbor_row.indices, neighbor_row.data):
            if rating >= 3 and column not in rated_columns:
                recommended_books.append(book_titles_cf[column])

    seen = set()
    final = []
    for book in recommended_books:
        if book not in seen:
            seen.add(book)
            final.append(book)

    return final[:5] or list(popular_df["title"].head(5).values)


def hybrid_recommend(username=None, user_id=None, book_name=None):
    if (username or user_id) and book_name:
        content_books = recommend(book_name)
        cf_books = recommend_cf(username=username, user_id=user_id)

        books_list = [book_name]
        for book in content_books + cf_books:
            if book != book_name and book not in books_list:
                books_list.append(book)

    elif username or user_id:
        books_list = recommend_cf(username=username, user_id=user_id)

    elif book_name:
        content_books = recommend(book_name)
        books_list = [book_name]
        for book in content_books:
            if book != book_name and book not in books_list:
                books_list.append(book)

    else:
        books_list = list(popular_df["title"].head(6).values)

    return [get_book_info(book) for book in books_list[:6]]
