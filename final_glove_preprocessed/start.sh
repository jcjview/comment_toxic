wget http://oybow8tnz.bkt.clouddn.com/cleaned-toxic-comments.zip ../data/
unzip -o ../data/cleaned-toxic-comments.zip  -d ../data/
python test_preprocess.py
python keras_pooled_gru_cnn1_cv_fold10.py
python keras_lstm_neptune_generate_cv_fold10.py
python keras_bgru_cv2_fold10.py

