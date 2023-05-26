import streamlit as st
import pandas as pd
import requests

# Define the available JSON files
json_files = {
    "Descriptive": "https://github.com/ShadmanRohan/BenCoref/raw/master/data/train/descriptive.json",
    "Novel": "https://github.com/ShadmanRohan/BenCoref/raw/master/data/train/novel.json",
    "Story": "https://github.com/ShadmanRohan/BenCoref/raw/master/data/train/story.json",
    "Biography": "https://github.com/ShadmanRohan/BenCoref/raw/master/data/train/biography.json"
}
st.header("BenCoref Explorer")

# Create a dropdown to select a JSON file
selected_file = st.selectbox("Select a file", list(json_files.keys()))

# Define the URL of the selected JSON file
json_url = json_files[selected_file]

# Fetch the JSON data
response = requests.get(json_url)
data = response.json()

# Convert the JSON data to a DataFrame
df = pd.DataFrame(data)

# Create a dropdown to select an ID
selected_id = st.selectbox("Select an ID", df['id'].unique())

# Filter the DataFrame based on the selected ID
selected_data = df[df['id'] == selected_id]['sentences'].tolist()
mention_clusters = df[df['id'] == selected_id]['mention_clusters'].tolist()

# Display the selected data in a formatted way
st.subheader("Data Point")
if len(selected_data) == 0:
    st.write("No data found for the selected ID.")
else:
    st.write(f"**ID: {selected_id}**")
    for idx, sentence in enumerate(selected_data[0]):
        st.write(idx)
        st.write(' '.join(sentence))
    st.write(f"**Clusters:**")
    for idx, sentence in enumerate(mention_clusters[0]):
        st.write(idx)
        st.write(str(mention_clusters[0][idx]))
#st.write(mention_clusters[0][1])