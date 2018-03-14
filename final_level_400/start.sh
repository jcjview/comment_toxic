wget https://kaggle2.blob.core.windows.net/forum-message-attachments/272289/8338/train_fr.csv ../data/
wget https://kaggle2.blob.core.windows.net/forum-message-attachments/272289/8339/train_es.csv ../data/
wget https://kaggle2.blob.core.windows.net/forum-message-attachments/272289/8340/train_de.csv ../data/
python test_preprocess.py
python keras_bgru_cv2_fold10.py
python keras_pooled_gru_cnn_cv_fold10.py
