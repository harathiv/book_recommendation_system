import pandas as pd
#1
# Load datasets (FIXED)
books = pd.read_csv("data/Books.csv", sep=";", encoding='ISO-8859-1', on_bad_lines='skip')
users = pd.read_csv("data/Users.csv", sep=";", encoding='ISO-8859-1', on_bad_lines='skip')
ratings = pd.read_csv("data/Ratings.csv", sep=";", encoding='ISO-8859-1', on_bad_lines='skip')
#2
# Display first few rows
#to check if the datasets are loaded correctly and to understand their structure
"""print("Books Data:")
print(books.head())

print("\nUsers Data:")
print(users.head())

print("\nRatings Data:")
print(ratings.head())
#3
#EDA (Exploratory Data Analysis)
# Shape of datasets
print("\nShape of Books:", books.shape)
print("Shape of Users:", users.shape)
print("Shape of Ratings:", ratings.shape)

# Column names
print("\nBooks Columns:", books.columns)
print("Users Columns:", users.columns)
print("Ratings Columns:", ratings.columns)

# Info about datasets
print("\nBooks Info:")
print(books.info())

print("\nUsers Info:")
print(users.info())

print("\nRatings Info:")
print(ratings.info())

# Check missing values
print("\nMissing values in Books:\n", books.isnull().sum())
print("\nMissing values in Users:\n", users.isnull().sum())
print("\nMissing values in Ratings:\n", ratings.isnull().sum())
"""
print(books.columns)
# Clean column names
books.columns = books.columns.str.strip().str.lower()

# Drop rows where Author or Publisher is missing
books = books.dropna(subset=['author', 'publisher'])

# Fill missing Age with average
# Convert Age to numeric (fix mixed data issue)
users['Age'] = pd.to_numeric(users['Age'], errors='coerce')

# Fill missing Age with mean
users['Age'] = users['Age'].fillna(users['Age'].mean())

# Check again
"""print("\nMissing values after cleaning:")

print("\nBooks:\n", books.isnull().sum())
print("\nUsers:\n", users.isnull().sum())
print("\nRatings:\n", ratings.isnull().sum())
"""

#as already book dataset was converted ,now we convert user and are dataset also into lower case
ratings.columns = ratings.columns.str.strip().str.lower()
users.columns = users.columns.str.strip().str.lower()


#4
#mergeing datasets
# Merge 'ratings' with 'books' as isbn is common
ratings_with_books = ratings.merge(books, on='isbn')
"""
print("\nMerged Data (Ratings + Books):")
print(ratings_with_books.head())

print("\nShape after merging ratings & books:", ratings_with_books.shape)

"""
# Merge with users (final dataset) new{combo of ratings and books} with 3rd dataset 'user'
final_data = ratings_with_books.merge(users, on='user-id')

"""
print("\nFinal merged data:")
print(final_data.head())

print("\nFinal dataset shape:", final_data.shape)
"""
#5 : remove inactive users and unpopular books
#filter users being active that rated movies
# Count number of ratings per user
user_counts = final_data['user-id'].value_counts()

# Keep users who rated more than 200 books
active_users = user_counts[user_counts > 20].index

filtered_data = final_data[final_data['user-id'].isin(active_users)]

print("\nAfter filtering users:", filtered_data.shape)

#filter books that has more than 50 ratings

# Count number of ratings per book
book_counts = filtered_data['title'].value_counts()

# Keep books with at least 50 ratings
popular_books = book_counts[book_counts >= 10].index

filtered_data = filtered_data[filtered_data['title'].isin(popular_books)]

print("\nAfter filtering books:", filtered_data.shape)

#6
# strong baseline model creation
#POPULARITY BASED RECOMMENDER SYSTEM

# Calculate number of ratings per book
num_ratings = filtered_data.groupby('title').count()['rating'].reset_index()
num_ratings.rename(columns={'rating': 'num_ratings'}, inplace=True)

# Calculate average rating per book
avg_ratings = filtered_data.groupby('title')['rating'].mean().reset_index()
avg_ratings.rename(columns={'rating': 'avg_rating'}, inplace=True)

# Merge both so we get table format of title,number of ratings,avg rating
popular_df = num_ratings.merge(avg_ratings, on='title')

#  ADD FILTER HERE
popular_df = popular_df[popular_df['num_ratings'] >= 50]


# Sort by highest rating
popular_df = popular_df.sort_values(by='avg_rating', ascending=False)

pd.set_option('display.max_columns', None)

"""print("\nTop Popular Books:")
print(popular_df.head(10))



print(final_data.columns)"""

#7
# COLLABORATIVE FILTERING BASED RECOMMENDER SYSTEM
# Remove ratings = 0 (not real ratings)
filtered_data = filtered_data[filtered_data['rating'] != 0]

# it creates book table with unique book titles and their corresponding authors
book_df = filtered_data[['title', 'author']].drop_duplicates()

#combine book title and author to create unique identifier for each book
book_df['features'] = book_df['title'] + " " + book_df['author']
# eg:"Harry Potter J.K. Rowling" with book and author


#convert text to vectors using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#create object  of countvectorizer as cv and fit transform the values
cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform(book_df['features'])


#measures how two books are similar based on their vector representations
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)


#recommendation function
def recommend(book_name):

    if not book_name:
        return []

    query = book_name.strip().lower()

    # reset index to match similarity matrix
    temp_df = book_df.reset_index(drop=True)

    matches = temp_df[temp_df['title'].str.lower() == query]

    if matches.empty:
        matches = temp_df[temp_df['title'].str.lower().str.contains(query, na=False)]

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


    




#for users login  testing 

from auth import register_user, login_user





#collaborative filtering based recommender system
#create user-item matrix

user_book_table = filtered_data.pivot_table(index='user-id', columns='title', values='rating')

#fillig missing values
user_book_table.fillna(0, inplace=True)

# calcualte similiarity between users

from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(user_book_table)

#convert into dataframe
import pandas as pd

user_similarity_df = pd.DataFrame(user_similarity, index=user_book_table.index, columns=user_book_table.index)

#recommendation function for collaborative filtering
def recommend_cf(user_id):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    
    recommended_books = []
    
    for similar_user in similar_users.index:
        books = user_book_table.loc[similar_user]
        liked_books = books[books > 3].index
        
        for book in liked_books:
            recommended_books.append(book)
    
    return list(set(recommended_books))[:5]

print(user_book_table.index[:10])
print(recommend_cf(6323))



#hybrid based model(popularity + collaborative filtering)

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

        books_list = []
        books_list.append(book_name)

        for b in content_books:
            if b != book_name and b not in books_list:
                books_list.append(b)

    else:
        books_list = list(popular_df['title'].values)

    final_output = []

    for book in books_list[:6]:
        book_info = books[books['title'] == book]

        if not book_info.empty:
            title = book_info.iloc[0]['title']
            author = book_info.iloc[0]['author']

            rating_row = popular_df[popular_df['title'] == book]
            rating = round(rating_row.iloc[0]['avg_rating'], 2) if not rating_row.empty else "N/A"

            final_output.append({
                "title": title,
                "author": author,
                "rating": rating
            })

    return final_output


    # list of unique book titles for UI
book_titles = sorted(book_df['title'].dropna().unique().tolist())

# Testing hybrid recommendations
# popularity (new user )
"""print(hybrid_recommend())

print(hybrid_recommend(user_id=6543))

print(hybrid_recommend(book_name="Harry Potter and the Sorcerer's Stone (Book 1)"))

#all mix (hybrid)
print(hybrid_recommend(user_id=6543, book_name="Harry Potter and the Sorcerer's Stone (Book 1)"))
"""

# export valid user ids
user_ids = list(user_book_table.index)
