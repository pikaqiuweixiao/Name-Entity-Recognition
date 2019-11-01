import numpy as np
from bert_lstm_ner import *
from tensorflow.python.saved_model import tag_constants

flags.DEFINE_string(
    "predict_file", FLAGS.output_dir,
    "The file of predict records")

processor = NerProcessor()
label_list = processor.get_labels()

with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.export_dir)
    op_list = graph.get_operations()
    for op in op_list:
        print(op.name)
    #     # print(op.values)
    tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
    print('tensor_input_ids:', tensor_input_ids)
    tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
    tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
    tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
    tensor_outputs = graph.get_tensor_by_name('blstm_crf_layer/concat:0')
    # tensor_logits = graph.get_tensor_by_name('project/Reshape:0')
    # trans = graph.get_tensor_by_name('crf_loss/transitions:0')
    print('tensor_outputs:', tensor_outputs)


def deep_ner(template):
    examples = []
    guid = '0'
    words = tokenization.convert_to_unicode(template)
    label_temp = ["O"] * len(words)
    labels = ' '.join([label for label in label_temp if len(label) > 0])
    text = ' '.join([word for word in words if len(word) > 0])
    print('text:', text)
    print('labels:', labels)
    examples.append(InputExample(guid=guid, text=text, label=labels))
    predict_examples = examples
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    filed_based_convert_examples_to_features(predict_examples, label_list,
                                             FLAGS.max_seq_length, tokenizer,
                                             predict_file, mode="test")

    record_iterator = tf.python_io.tf_record_iterator(path=predict_file)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        input_ids = example.features.feature['input_ids'].int64_list.value
        input_mask = example.features.feature['input_mask'].int64_list.value
        label_ids = example.features.feature['label_ids'].int64_list.value
        segment_ids = example.features.feature['segment_ids'].int64_list.value
        result = sess.run(tensor_outputs, feed_dict={
            tensor_input_ids: np.array(input_ids).reshape(-1, FLAGS.max_seq_length),
            tensor_input_mask: np.array(input_mask).reshape(-1, FLAGS.max_seq_length),
            tensor_label_ids: np.array(label_ids).reshape(-1, FLAGS.max_seq_length),
            tensor_segment_ids: np.array(segment_ids).reshape(-1, FLAGS.max_seq_length),
        })
        print(result)
        prediction = result[0]
        result_labels = []
        for id in prediction:
            if id != 0:
                result_labels.append(id2label[id])
        result_labels.reverse()
        result_labels.remove("[SEP]")
        result_labels.remove("[CLS]")
    return result_labels


if __name__ == "__main__":
    print(deep_ner('中国在亚洲的什么位置？'))
