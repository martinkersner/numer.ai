import collections
import tensorflow as tf

training_data = "numerai_training_data.csv"
tournament_data = "numerai_tournament_data.csv"

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


def dataset_input_fn():
    training_data = ["numerai_training_data.csv"]
    # dataset = tf.data.TextLineDataset(training_data).skip(1)
    dataset = tf.data.TextLineDataset(training_data).skip(53571)

    def _decode_line(line):
        items = tf.decode_csv(line, list(defaults.values()))
        pairs = zip(defaults.keys(), items)
        features_dict = dict(pairs)
        label = features_dict.pop("target")
        return features_dict, label

    dataset = dataset.map(_decode_line)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels


feature_columns = [tf.contrib.layers.real_valued_column(k) for k in defaults.keys() if "feature" in k]
estimator = tf.estimator.DNNRegressor(
    feature_columns=feature_columns,
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.AdamOptimizer(),
)

print("Training")
estimator.train(input_fn=dataset_input_fn,
                max_steps=10)

print("Prediction")
for i in estimator.predict(input_fn=dataset_input_fn):
    print(i)

print("Evaluation")
result = estimator.evaluate(input_fn=dataset_input_fn,
                            steps=2)

print(result)
