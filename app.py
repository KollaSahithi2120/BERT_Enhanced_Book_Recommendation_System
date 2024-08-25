import streamlit as st
import sqlite3
import pandas as pd
import pickle
import torch


# Load BERT model and tokenizer (use st.cache_resource to cache the model)
@st.cache_resource
def load_bert_model():
    try:
        with open("bert_model.pkl", "rb") as file:  # Adjust path as needed
            bert_model = pickle.load(file)
        return bert_model
    except Exception as e:
        st.error(f"Failed to load BERT model: {e}")
        return None

# Database connection
def get_db_connection():
    conn = sqlite3.connect('books.db')  # Ensure 'books.db' is in the same directory as this script
    return conn

# Fetch books from the database
def fetch_books(search_query=""):
    conn = get_db_connection()
    query = "SELECT * FROM books WHERE book_name LIKE ? OR author_name LIKE ? OR genre LIKE ?"
    books = pd.read_sql(query, conn, params=[f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'])
    conn.close()
    return books

# Add a book to a list (wishlist or cart)
def add_to_list(book_id, list_type):
    if list_type not in st.session_state:
        st.session_state[list_type] = []
    if book_id not in st.session_state[list_type]:
        st.session_state[list_type].append(book_id)

# Display books with heart and cart buttons
def display_books(books):
    for i in range(0, len(books), 2):
        cols = st.columns(2)

        for j in range(2):
            if i + j < len(books):
                row = books.iloc[i + j]
                with cols[j]:
                    st.image(row['image_url'], width=150)
                    st.write(f"**{row['book_name']}**")
                    st.write(f"*{row['author_name']}*")
                    st.write(f"**Genre**: {row['genre']}")
                    st.write(f"**Published**: {row['yop']}")
                    st.write(f"**Publisher**: {row['name_of_publisher']}")

                    # Heart for Wishlist
                    if st.button(f"❤️ Add to Wishlist", key=f"wishlist_{row['id']}"):
                        add_to_list(row['id'], 'wishlist')
                        st.success("Added to Wishlist")

                    # Button for Cart
                    if st.button("🛒 Add to Cart", key=f"cart_{row['id']}"):
                        add_to_list(row['id'], 'cart')
                        st.success("Added to Cart")

                    # View Details Button
                    if st.button("View Details", key=f"view_{row['id']}"):
                        st.write(row['description'])
            st.write("---")



# View books in a list (wishlist or cart)
def view_list(list_type):
    if list_type in st.session_state and st.session_state[list_type]:
        books = fetch_books()
        list_books = books[books['id'].isin(st.session_state[list_type])]
        display_books(list_books)
    else:
        st.write("Your list is empty.")

# Recommend books using BERT model
def recommend_books_bert(book_names):
    bert_model = load_bert_model()
    if bert_model is None:
        st.write("BERT model not loaded, unable to generate recommendations.")
        return pd.DataFrame()  # Return an empty DataFrame

    books = fetch_books()

    # Placeholder logic for recommendations using the BERT model
    recommendations = set()
    for book_name in book_names:
        # Assuming bert_model is a dictionary of book_name to recommended_book_ids
        if book_name in bert_model:
            recommendations.update(bert_model[book_name])

    # Remove books that are already in the wishlist
    if 'wishlist' in st.session_state:
        recommendations.difference_update(st.session_state['wishlist'])

    return books[books['id'].isin(recommendations)]

# Home page displaying all books
def home_page():
    st.title("Book Recommendation System")
    books = fetch_books()
    display_books(books)

# Wishlist page displaying books added to wishlist
def wishlist_page():
    st.title("Wishlist")
    view_list('wishlist')

# Cart page displaying books added to cart
def cart_page():
    st.title("Cart")
    view_list('cart')

# Search page to search for books
def search_page():
    st.title("Search Books")
    search_query = st.text_input("Enter search term...")
    if search_query:
        books = fetch_books(search_query)
        display_books(books)
    else:
        st.write("Please enter a search term.")

# Recommended books page
def recommended_page():
    st.title("Recommended Books")
    if 'wishlist' in st.session_state and st.session_state['wishlist']:
        books = fetch_books()
        wishlist_books = books[books['id'].isin(st.session_state['wishlist'])]

        # Extract book names and generate recommendations
        book_names = wishlist_books['book_name'].tolist()
        recommended_books = recommend_books_bert(book_names)

        if not recommended_books.empty:
            st.write("Based on your wishlist, we recommend the following books:")
            display_books(recommended_books)
        else:
            st.write("No recommendations found.")
    else:
        st.write("Add books to your wishlist to get recommendations.")

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Search", "Wishlist", "Cart", "Recommended"])

    if page == "Home":
        home_page()
    elif page == "Search":
        search_page()
    elif page == "Wishlist":
        wishlist_page()
    elif page == "Cart":
        cart_page()
    elif page == "Recommended":
        recommended_page()

if __name__ == "__main__":
    main()
