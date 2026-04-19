import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def prepare_model():
    # Load datasets
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

    # Clean column names
    books.columns = books.columns.str.strip().str.lower()
    users.columns = users.columns.str.strip().str.lower()
    ratings.columns = ratings.columns.str.strip().str.lower()
   

    # FIX datatype mismatch
    users['user-id'] = pd.to_numeric(users['user-id'], errors='coerce')
    ratings['user-id'] = pd.to_numeric(ratings['user-id'], errors='coerce')

# drop invalid rows
    users = users.dropna(subset=['user-id'])
    ratings = ratings.dropna(subset=['user-id'])

# convert to int
    users['user-id'] = users['user-id'].astype(int)
    ratings['user-id'] = ratings['user-id'].astype(int)
    # Clean missing values
    books = books.dropna(subset=["author", "publisher"])
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users["age"] = users["age"].fillna(users["age"].mean())

    # Merge datasets
    ratings_with_books = ratings.merge(books, on="isbn")
    final_data = ratings_with_books.merge(users, on="user-id")

    # Filter users and books
    user_counts = final_data["user-id"].value_counts()
    active_users = user_counts[user_counts > 20].index
    filtered_data = final_data[final_data["user-id"].isin(active_users)]

    book_counts = filtered_data["title"].value_counts()
    popular_books = book_counts[book_counts >= 10].index
    filtered_data = filtered_data[filtered_data["title"].isin(popular_books)]

    # Popularity model
    num_ratings = filtered_data.groupby("title").count()["rating"].reset_index()
    num_ratings.rename(columns={"rating": "num_ratings"}, inplace=True)

    avg_ratings = filtered_data.groupby("title")["rating"].mean().reset_index()
    avg_ratings.rename(columns={"rating": "avg_rating"}, inplace=True)

    popular_df = num_ratings.merge(avg_ratings, on="title")
    popular_df = popular_df[popular_df["num_ratings"] >= 10]
    popular_df = popular_df.sort_values(by="avg_rating", ascending=False)

    # Remove implicit ratings
    filtered_data = filtered_data[filtered_data["rating"] != 0]

    # Content-based prep
    book_df = filtered_data[["title", "author"]].drop_duplicates().reset_index(drop=True)
    book_df["features"] = book_df["title"] + " " + book_df["author"]

    cv = CountVectorizer(stop_words="english")
    vectors = cv.fit_transform(book_df["features"])
    similarity = cosine_similarity(vectors)

    # Collaborative filtering prep
    user_book_table = filtered_data.pivot_table(index="user-id", columns="title", values="rating")
    user_book_table.fillna(0, inplace=True)

    user_similarity = cosine_similarity(user_book_table)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_book_table.index,
        columns=user_book_table.index
    )

    book_titles = sorted(book_df["title"].dropna().unique().tolist())
    user_ids = list(user_book_table.index)

    return books, popular_df, book_df, similarity, user_book_table, user_similarity_df, book_titles, user_ids


# Prepare once and cache
books, popular_df, book_df, similarity, user_book_table, user_similarity_df, book_titles, user_ids = prepare_model()


def recommend(book_name):
    if not book_name:
        return []

    query = book_name.strip().lower()
    temp_df = book_df.reset_index(drop=True)

    matches = temp_df[temp_df["title"].str.lower() == query]

    if matches.empty:
        matches = temp_df[temp_df["title"].str.lower().str.contains(query, na=False)]

    if matches.empty:
        return []

    book_index = matches.index[0]
    distances = similarity[book_index]

    books_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommended_books = []
    for i in books_list:
        recommended_books.append(temp_df.iloc[i[0]].title)

    return recommended_books


def recommend_cf(user_id):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]

    recommended_books = []

    for similar_user in similar_users.index:
        user_books = user_book_table.loc[similar_user]
        liked_books = user_books[user_books >= 3].index

        for book in liked_books:
            recommended_books.append(book)

    return list(set(recommended_books))[:5]


def hybrid_recommend(user_id=None, book_name=None):
    if user_id and book_name:
        content_books = recommend(book_name)
        cf_books = recommend_cf(user_id)

        books_list = []
        if book_name not in books_list:
            books_list.append(book_name)

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
        books_list = list(popular_df["title"].values)

    final_output = []

    for book in books_list[:6]:
        book_info = books[books["title"] == book]

        if not book_info.empty:
            title = book_info.iloc[0]["title"]
            author = book_info.iloc[0]["author"]

            rating_row = popular_df[popular_df["title"] == book]
            rating = round(rating_row.iloc[0]["avg_rating"], 2) if not rating_row.empty else "N/A"

            final_output.append({
                "title": title,
                "author": author,
                "rating": rating
            })

    return final_output