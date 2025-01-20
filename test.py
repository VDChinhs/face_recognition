import pickle
import faiss
from modules.processors.face_analyser import get_many_faces
import cv2
import numpy as np

img = cv2.imread('inputs/chinh.jpg')
faces = get_many_faces(img)

em = faces[0].embedding
# print(em)

with open('database\embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

filenames = data['filenames']
embeddings = data['embeddings']
print(embeddings)

database_embeddings = np.array(embeddings).astype('float32')

# print(database_embeddings)

faiss.normalize_L2(database_embeddings)
index = faiss.IndexFlatIP(512)
index.add(database_embeddings)

k = 1
query_embedding = np.expand_dims(faces[0].embedding, axis=0)
faiss.normalize_L2(query_embedding)
distances, indices = index.search(query_embedding, k)


confident = (1 + distances[0][0]) / 2
print(f"Name: {filenames[indices[0][0]]}, Confidence Score: {confident}")




# for score, idx in zip(distances[0], indices[0]):
#     confident = (1 + score) / 2
#     print(f"Name: {filenames[idx]}, Confidence Score: {confident}")