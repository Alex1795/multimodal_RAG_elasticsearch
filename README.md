# Multimodal RAG with Elasticseach and geoqueries 🏔️

This project was created for a blog for Elasticsearch. It takes the implementation of a multimodal RAG system to build an assistant that recommends activities in national parks based on a RAG search against Elasticsearch. 

Blog link: <TO DO>

The project covers all the parts needed to create the data, index it on Elastic and run a front end to search and interact with the assistant.

## System architecture
### Indexing pipeline
It will handle the vectorization of the image and description, add more metadata of the image to create a document and index it to Elastic:

### Search pipeline

This stage will handle the user’s query, create the search and generate a response from the search results. The LLM used is cogito:3b with Ollama, though it would be easily replaced by any remote model (like Claude or ChatGPT). We chose this particular model because it’s lightweight and it excels at general tasks (as is expected from an assistant) compared to similar models (like llama 3.2 3b). This means we get good results without long wait times and everything is running locally!


### Web Application
A front end based on streamlit that handles the user input and runs the search pipeline to obtain the LLM final response and displays images from the search results with their descriptions.



## How to use 

### You need:

- An Elastic deployment
- Ollama running locally and the model cogito:3b running
- To copy this directory locally


### Indexing:
- Run the upload_documents.py file, it will take the images and metadata from the images, generate the necessary embeddings and upload all the data to Elastic.

Simply run:
```
python upload_documents.py
```
### Search:

You can run the LLM_conversation.py file to get the responses in the console

You can do this by running:
```
python LLM_conversation.py
```


Or simply run the streamlit app and use the UI in your browser using:

```
streamlit run streamlit_app.py
```
