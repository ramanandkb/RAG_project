from langchain_huggingface import HuggingFaceEmbeddings

text="I"
model=HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
vector=model.embed_query(text)

print(vector)
print(len(vector))
