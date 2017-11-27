import sys
import collections
import tensorflow as tf
import argparse

defaults = collections.OrderedDict([
    ("id", [""]),
    ("era", [""]),
    ("data_type", [""]),
    ("feature1", [0.0]),
    ("feature2", [0.0]),
    ("feature3", [0.0]),
    ("feature4", [0.0]),
    ("feature5", [0.0]),
    ("feature6", [0.0]),
    ("feature7", [0.0]),
    ("feature8", [0.0]),
    ("feature9", [0.0]),
    ("feature10", [0.0]),
    ("feature11", [0.0]),
    ("feature12", [0.0]),
    ("feature13", [0.0]),
    ("feature14", [0.0]),
    ("feature15", [0.0]),
    ("feature16", [0.0]),
    ("feature17", [0.0]),
    ("feature18", [0.0]),
    ("feature19", [0.0]),
    ("feature20", [0.0]),
    ("feature21", [0.0]),
    ("feature22", [0.0]),
    ("feature23", [0.0]),
    ("feature24", [0.0]),
    ("feature25", [0.0]),
    ("feature26", [0.0]),
    ("feature27", [0.0]),
    ("feature28", [0.0]),
    ("feature29", [0.0]),
    ("feature30", [0.0]),
    ("feature31", [0.0]),
    ("feature32", [0.0]),
    ("feature33", [0.0]),
    ("feature34", [0.0]),
    ("feature35", [0.0]),
    ("feature36", [0.0]),
    ("feature37", [0.0]),
    ("feature38", [0.0]),
    ("feature39", [0.0]),
    ("feature40", [0.0]),
    ("feature41", [0.0]),
    ("feature42", [0.0]),
    ("feature43", [0.0]),
    ("feature44", [0.0]),
    ("feature45", [0.0]),
    ("feature46", [0.0]),
    ("feature47", [0.0]),
    ("feature48", [0.0]),
    ("feature49", [0.0]),
    ("feature50", [0.0]),
    ("target", [0])
])  # pyformat: disable


tf.logging.set_verbosity(tf.logging.INFO)

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def _decode_line(line):
        items = tf.decode_csv(line, list(defaults.values()))
        pairs = zip(defaults.keys(), items)
        features_dict = dict(pairs)
        label = features_dict.pop("target")
        return features_dict, label

    dataset = tf.data.TextLineDataset(file_path).skip(1)
    dataset = dataset.map(_decode_line)

    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def process_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data", type=str, default="numerai_training_data.csv", help="Path to the training data.")
    parser.add_argument(
        "--tournament_data", type=str, default="numerai_tournament_data.csv", help="Path to the test data.")
    return parser.parse_args()


def model_fn(features, labels, mode, params):
    feature_columns = [tf.feature_column.numeric_column(k) for k in defaults.keys() if "feature" in k]
    input_layer = tf.feature_column.input_layer(
        features=features, feature_columns=feature_columns)

    second_hidden_layer = tf.layers.dense(input_layer, 10, activation=tf.nn.relu)

    output_layer = tf.layers.dense(second_hidden_layer, 1)

    predictions = tf.reshape(output_layer, [-1])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"target": predictions})

    # loss = tf.losses.log_loss(labels, predictions)
    loss = tf.losses.mean_squared_error(labels, predictions)

    eval_metric_ops = {
    "rmse": tf.metrics.root_mean_squared_error(
        tf.cast(labels, tf.float32), predictions)
    }

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)



def main():
    args = process_arguments()
    FILE_TRAIN = args.training_data
    FILE_TEST = args.tournament_data

    FILE_TRAIN = "numerai_training_data_head.csv"
    FILE_TEST = "numerai_training_data_head.csv"

    model_params = {"learning_rate": 0.001}
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

    nn.train(
        input_fn=lambda: my_input_fn(FILE_TRAIN, True, 1), steps=2000)

    nn.evaluate(
        input_fn=lambda: my_input_fn(FILE_TRAIN, False, 1))

    predictions = nn.predict(
        input_fn=lambda: my_input_fn(FILE_TRAIN, False, 1))

    for prediction in predictions:
        print(prediction["target"])

if __name__ == "__main__":
    main()

# def tmp(fFILE_TRAIN, FILE_TEST):
    # feature_columns = [tf.feature_column.numeric_column(k) for k in defaults.keys() if "feature" in k]
    # estimator = tf.estimator.DNNRegressor(
        # feature_columns=feature_columns,
        # hidden_units=[10, 10],
        # optimizer=tf.train.AdamOptimizer(),
        # model_dir="logdir")

    # estimator.train(
        # input_fn=lambda: my_input_fn(FILE_TRAIN, True, 1))

    # evaluate_result = estimator.evaluate(
        # input_fn=lambda: my_input_fn(FILE_TEST, False, 1))

    # for key in evaluate_result:
        # print("   {}: {}".format(key, evaluate_result[key]))

    # predict_results = estimator.predict(
        # input_fn=lambda: my_input_fn(FILE_TEST, False, 1))

    # for prediction in predict_results:
        # print(prediction["predictions"][0])
