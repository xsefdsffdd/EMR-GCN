import tensorflow as tf
from sklearn import metrics

def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss without masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)


def softmax_accuracy(preds, labels,name):
    """Accuracy without masking."""
    preds = tf.nn.softmax(preds)
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all,name=name)


# compute scores
def compute_scores(test_labels, test_pred):
    print("Test Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_labels, test_pred, digits=4))
    print("Macro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
    print("Micro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))