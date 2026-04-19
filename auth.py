import sqlite3

def register_user(username, password):
    try:
        with sqlite3.connect("users.db", check_same_thread=False) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT UNIQUE,
                password TEXT
            )
            """)
            
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            
            conn.commit()
            
            return True
    
    except sqlite3.IntegrityError:
        return False   # username already exists


def login_user(username, password):
    with sqlite3.connect("users.db", check_same_thread=False) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT,
            password TEXT
        )
        """)
        
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        
        result = cursor.fetchone()
        
        return result



