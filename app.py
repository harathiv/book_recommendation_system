import random
import streamlit as st
from auth import login_user, register_user
from main import hybrid_recommend, book_titles, user_ids

st.set_page_config(
    page_title="Book Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #312e81);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1f2937);
}

.main-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
    margin-bottom: 10px;
}

.sub-title {
    font-size: 30px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 10px;
    margin-bottom: 20px;
}

.auth-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.08);
}

.rec-card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 14px;
    border-left: 6px solid #38bdf8;
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}

.selected-card {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 14px;
    color: white;
    box-shadow: 0 8px 20px rgba(37,99,235,0.35);
}

.reason-box {
    background: rgba(59,130,246,0.18);
    border: 1px solid rgba(96,165,250,0.35);
    padding: 14px 18px;
    border-radius: 14px;
    color: #dbeafe;
    margin-bottom: 18px;
    font-size: 18px;
    font-weight: 600;
}

.section-title {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 20px;
    margin-bottom: 15px;
}

.small-note {
    color: #cbd5e1;
    font-size: 15px;
}

div.stButton > button {
    background: linear-gradient(90deg, #ec4899, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    font-size: 17px;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #db2777, #7c3aed);
    color: white;
}

.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    border-radius: 12px !important;
}

.sidebar-title {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 10px;
}

.welcome-box {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    padding: 14px;
    border-radius: 14px;
    color: white;
    font-weight: 700;
    font-size: 18px;
    margin-top: 12px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="main-title">📚 Book Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#cbd5e1; font-size:18px;">Discover books through popularity, similarity, personalization, and hybrid intelligence</p>',
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
st.sidebar.markdown('<div class="sidebar-title">✨ Navigation</div>', unsafe_allow_html=True)
menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

# ---------- LOGIN ----------
if choice == "Login" and 'user' not in st.session_state:
    st.markdown('<div class="sub-title">🔐 Login</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.session_state["user"] = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- REGISTER ----------
elif choice == "Register" and 'user' not in st.session_state:
    st.markdown('<div class="sub-title">📝 Create Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    new_user = st.text_input("Create Username")
    new_password = st.text_input("Create Password", type="password")

    if st.button("Register"):
        success = register_user(new_user, new_password)
        if success:
            st.success("Account created successfully")
        else:
            st.error("Username already exists")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- AFTER LOGIN ----------
if "user" in st.session_state:
    st.sidebar.markdown(
        f'<div class="welcome-box">Welcome {st.session_state["user"]}</div>',
        unsafe_allow_html=True
    )

    if st.sidebar.button("Logout"):
        del st.session_state["user"]
        st.rerun()

    st.markdown('<div class="sub-title">🎯 Get Book Recommendations</div>', unsafe_allow_html=True)

    option = st.selectbox(
        "Choose Recommendation Type",
        ["Popular", "By Book Name", "Personalized", "Hybrid"]
    )

    book_name = st.selectbox("Select Book (optional)", [""] + book_titles)

    result = None
    reason = ""

    if st.button("Recommend"):
        uid = random.choice(user_ids) if user_ids else None
        bname = book_name if book_name else None

        if option in ["By Book Name", "Hybrid"] and not bname:
            st.error("Please select a book.")
            st.stop()

        if option == "Popular":
            result = hybrid_recommend()
            reason = "📊 Based on overall popularity and high ratings"

        elif option == "By Book Name":
            result = hybrid_recommend(book_name=bname)
            reason = f"🔎 First showing your selected book, then books similar to '{bname}'"

        elif option == "Personalized":
            result = hybrid_recommend(user_id=uid)
            reason = "👤 Based on users with similar reading interests"

        elif option == "Hybrid":
            result = hybrid_recommend(user_id=uid, book_name=bname)
            reason = "⚡ First showing your selected book, then a mix of similar and personalized recommendations"

        if not result:
            st.error("No recommendations found.")
        else:
            st.markdown('<div class="section-title">📚 Recommended Books</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

            for i, book in enumerate(result):
                search_query = book["title"].replace(" ", "+")
                url = f"https://www.google.com/search?q={search_query}"

                if i == 0 and option in ["By Book Name", "Hybrid"]:
                    st.markdown(f"""
                    <div class="selected-card">
                        <h3>✅ Selected Book</h3>
                        <p style="font-size:22px; font-weight:700;">📖 {book['title']}</p>
                        <p style="font-size:18px;">✍️ Author: {book['author']}</p>
                        <p style="font-size:18px;">⭐ Rating: {book['rating']}</p>
                        <p><a href="{url}" target="_blank" style="color:white; font-weight:700;">🔗 View Book Online</a></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="rec-card">
                        <p style="font-size:21px; font-weight:700;">📖 {book['title']}</p>
                        <p style="font-size:17px;">✍️ Author: {book['author']}</p>
                        <p style="font-size:17px;">⭐ Rating: {book['rating']}</p>
                        <p><a href="{url}" target="_blank" style="color:#7dd3fc; font-weight:700;">🔗 View Book Online</a></p>
                    </div>
                    """, unsafe_allow_html=True)