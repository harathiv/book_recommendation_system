import hashlib
import hmac
import secrets
import sqlite3


DATABASE = "users.db"


def _hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()
    return f"{salt}${password_hash}"


def _verify_password(password, stored_password):
    if "$" not in stored_password:
        return hmac.compare_digest(password, stored_password)

    salt, expected_hash = stored_password.split("$", 1)
    actual_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()
    return hmac.compare_digest(actual_hash, expected_hash)


def _initialize_database(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_ratings (
            username TEXT NOT NULL,
            book_title TEXT NOT NULL,
            rating REAL NOT NULL CHECK (rating >= 1 AND rating <= 10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (username, book_title),
            FOREIGN KEY (username) REFERENCES users(username)
        )
        """
    )


def register_user(username, password):
    username = username.strip()
    if not username or not password:
        return False

    try:
        with sqlite3.connect(DATABASE, check_same_thread=False) as conn:
            _initialize_database(conn)
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, _hash_password(password)),
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False


def login_user(username, password):
    username = username.strip()
    with sqlite3.connect(DATABASE, check_same_thread=False) as conn:
        _initialize_database(conn)
        row = conn.execute(
            "SELECT password FROM users WHERE username=?",
            (username,),
        ).fetchone()

        if not row or not _verify_password(password, row[0]):
            return False

        # Migrate older plaintext records after a successful login.
        if "$" not in row[0]:
            conn.execute(
                "UPDATE users SET [password] = ? WHERE username = ?",
                (_hash_password(password), username),
            )
            conn.commit()

        return True


def save_rating(username, book_title, rating):
    with sqlite3.connect(DATABASE, check_same_thread=False) as conn:
        _initialize_database(conn)
        conn.execute(
            """
            INSERT INTO user_ratings (username, book_title, rating)
            VALUES (?, ?, ?)
            ON CONFLICT(username, book_title)
            DO UPDATE SET rating=excluded.rating,
                          created_at=CURRENT_TIMESTAMP
            """,
            (username, book_title, float(rating)),
        )
        conn.commit()


def get_all_user_ratings():
    with sqlite3.connect(DATABASE, check_same_thread=False) as conn:
        _initialize_database(conn)
        return conn.execute(
            "SELECT username, book_title, rating FROM user_ratings"
        ).fetchall()


def get_user_ratings(username):
    with sqlite3.connect(DATABASE, check_same_thread=False) as conn:
        _initialize_database(conn)
        return conn.execute(
            """
            SELECT book_title, rating
            FROM user_ratings
            WHERE username=?
            ORDER BY created_at DESC
            """,
            (username,),
        ).fetchall()
