############# Char level RNN ###################

MAX_DOC_LENGTH=100 

python text_classification_TPU_withTFRecords.py \
 --use_tpu=True \
 --model_dir=.gs://vinh-tutorial/output/charRNN/${MAX_DOC_LENGTH}\
 --train_batch_size=1024 \
 --num_cores=8 \
 --learning_rate=0.1 \
 --data_dir=gs://vinh-tutorial/data/dbpedia/${MAX_DOC_LENGTH} \
 --rnn_size=32 \
 --max_document_length=${MAX_DOC_LENGTH} \
 --master=grpc://10.240.22.2:8470


capture_tpu_profile \
--service_addr=10.240.22.2:8470 \
--logdir=gs://vinh-tutorial/output/charRNN/${MAX_DOC_LENGTH} \
--duration=10000



############# Word level RNN ###################
#data prep
MAX_DOC_LENGTH=50 

python dbpedia_to_TFRecords_word.py --max_document_length=${MAX_DOC_LENGTH}

gsutil cp word-train.tfrecords gs://vinh-tutorial/data/dbpedia/wordRNN/${MAX_DOC_LENGTH}/
gsutil cp word-test.tfrecords gs://vinh-tutorial/data/dbpedia/wordRNN/${MAX_DOC_LENGTH}/

python wordRNN_classification_TPU_withTFRecords.py \
 --use_tpu=True \
 --model_dir=.gs://vinh-tutorial/output/wordRNN/${MAX_DOC_LENGTH}\
 --train_batch_size=1024 \
 --num_cores=8 \
 --learning_rate=0.1 \
 --data_dir=gs://vinh-tutorial/data/dbpedia/wordRNN/${MAX_DOC_LENGTH} \
 --rnn_size=32 \
 --max_document_length=${MAX_DOC_LENGTH} \
 --master=grpc://10.240.22.2:8470