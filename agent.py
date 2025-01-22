import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import os
import json
from datetime import datetime, timedelta

# Set up the Streamlit App
st.title("AI Customer Support Agent with Memory 🛒")
st.caption("Chat with a customer support assistant who remembers your past interactions.")

# Load Hugging Face model and tokenizer
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"  # Pre-trained conversational model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Load SentenceTransformer for memory retrieval
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # For semantic similarity

similarity_model = load_similarity_model()

# In-memory storage for customer data and interactions
if "memory" not in st.session_state:
    st.session_state.memory = {}

# Generate synthetic customer data
def generate_synthetic_data(user_id):
    today = datetime.now()
    order_date = (today - timedelta(days=10)).strftime("%B %d, %Y")
    expected_delivery = (today + timedelta(days=2)).strftime("%B %d, %Y")
    customer_data = {
        "customer_id": user_id,
        "name": "John Doe",
        "email": "john.doe@example.com",
        "recent_order": {
            "order_id": "12345",
            "product": "High-End Laptop",
            "price": "$1500",
            "order_date": order_date,
            "expected_delivery": expected_delivery,
        },
        "previous_orders": [
            {"order_id": "12340", "product": "Smartphone", "price": "$700"},
            {"order_id": "12335", "product": "Noise Cancelling Headphones", "price": "$200"},
        ],
        "preferences": ["electronics", "gaming", "gadgets"],
    }
    st.session_state.memory[user_id] = {
        "profile": customer_data,
        "interactions": [],
    }
    return customer_data

# Retrieve relevant memories
def get_relevant_memories(query, user_id):
    if user_id not in st.session_state.memory:
        return []
    interactions = st.session_state.memory[user_id].get("interactions", [])
    if not interactions:
        return []
    query_embedding = similarity_model.encode(query, convert_to_tensor=True)
    memories = []
    for interaction in interactions:
        interaction_embedding = similarity_model.encode(interaction["query"], convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, interaction_embedding).item()
        if score > 0.6:  # Adjust threshold for relevance
            memories.append(interaction["query"])
    return memories

# Handle a user query
def handle_query(query, user_id):
    if user_id not in st.session_state.memory:
        st.error("No customer data found. Generate synthetic data first.")
        return "Please generate synthetic customer data before proceeding."
    relevant_memories = get_relevant_memories(query, user_id)
    context = " ".join(relevant_memories)
    input_text = f"{context} Customer: {query} Support Agent:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=300, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    # Save the interaction
    st.session_state.memory[user_id]["interactions"].append({"query": query, "response": response})
    return response

# Sidebar for customer ID and actions
st.sidebar.title("Customer Actions")
customer_id = st.sidebar.text_input("Enter your Customer ID")

if customer_id:
    if st.sidebar.button("Generate Synthetic Data"):
        customer_data = generate_synthetic_data(customer_id)
        st.sidebar.success("Synthetic data generated!")
        st.sidebar.json(customer_data)

    if st.sidebar.button("View Customer Profile"):
        if customer_id in st.session_state.memory:
            profile = st.session_state.memory[customer_id].get("profile", {})
            st.sidebar.json(profile)
        else:
            st.sidebar.warning("No profile found. Generate synthetic data first.")

    if st.sidebar.button("View Memory"):
        if customer_id in st.session_state.memory:
            interactions = st.session_state.memory[customer_id].get("interactions", [])
            if interactions:
                st.sidebar.write(f"Memory for customer {customer_id}:")
                for interaction in interactions:
                    st.sidebar.write(f"- {interaction['query']}: {interaction['response']}")
            else:
                st.sidebar.info("No memory found yet.")
        else:
            st.sidebar.warning("No memory found. Generate synthetic data first.")

# Chat interface
if customer_id:
    st.write(f"Chat with the support agent for Customer ID: {customer_id}")
    query = st.chat_input("How can I assist you today?")
    if query:
        response = handle_query(query, customer_id)
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(response)
else:
    st.warning("Please enter a Customer ID to start.")

