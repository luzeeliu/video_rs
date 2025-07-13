from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

dataset_path = os.path.join('..',"dataset")
train_path = os.path.join(dataset_path,"train","train.parquet")
val_path = os.path.join(dataset_path,"validation","val.parquet")

df_train = pd.read_parquet(train_path)
df_val = pd.read_parquet(val_path)

x_train = df_train['caption'].dropna().tolist()
x_val= df_val['caption'].dropna().tolist()

le = LabelEncoder().fit(df_train['coarse_label'])
y_train = le.transform(df_train['coarse_label'])
y_val   = le.transform(df_val  ['coarse_label'])

embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',device="cuda")
text_train = embedder.encode(x_train, batch_size=64, show_progress_bar= True)
text_val = embedder.encode(x_val, batch_size=64, show_progress_bar=True)

clf = LogisticRegression(
    solver='lbfgs',
    multi_class="multinomial",
    max_iter= 1000,
    class_weight="balanced"
)

clf.fit(text_train,y_train)

pred = clf.predict(text_val)
acc = accuracy_score(y_val, pred)
print(f"logreg baseline accuracy :{acc:.4f}")