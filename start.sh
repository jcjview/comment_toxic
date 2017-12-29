tar -xzvf data/train_corps.tar.gz  -C data/
tar -xzvf data/test_corps.tar.gz -C data/

cat data/train_corps.txt data/test_corps.txt > all.txt

python d2v.py
python ex2_txt2h5.py
python ex3_embeding_prepare.py
python keras_CNN_RNN_w2v.py