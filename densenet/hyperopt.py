from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import hp

from sklearn.metrics import accuracy_score
import csv, pickle, traceback
import tensorflow as tf

from dense_block import dense_block, transition_block

train_directory = "../dataset_normalized/training_set"
test_directory = "../dataset_normalized/validation_set"

train_gen = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    labels="inferred",
    label_mode="categorical",
    batch_size=3,
    image_size=(355, 370),
    validation_split=0.2,
    seed=2222,
    subset="training",
)

valid_gen = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    labels="inferred",
    label_mode="categorical",
    batch_size=8,
    image_size=(355, 370),
    validation_split=0.2,
    seed=2222,
    subset="validation",
)

test_gen = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    labels="inferred",
    label_mode="categorical",
    batch_size=8,
    image_size=(355, 370)
)

def pooling_choice(tensor, pooling_type: str):

    if pooling_type == "glob_avg":
        tensor = tf.keras.layers.GlobalAveragePooling2D() (tensor)
    elif pooling_type == "glob_max":
        tensor = tf.keras.layers.GlobalMaxPooling2D() (tensor)
    elif pooling_type == "avg":
        tensor = tf.keras.layers.AveragePooling2D() (tensor)
    elif pooling_type == "max":
        tensor = tf.keras.layers.MaxPooling2D() (tensor)
    
    tensor = tf.keras.layers.Flatten() (tensor)
    return tensor

def dense_choice(tensor, qtd_dense: int, filters: list, activation: str, dropout_value: int):
    for i in range(qtd_dense):
        tensor = tf.keras.layers.Dense(filters[i], activation=activation) (tensor)
        tensor = tf.keras.layers.Dropout(dropout_value) (tensor)
    
    return tensor

def dense_block_change_choice(tensor, qtd_denses, blocks):
    conv_numbers = 6
    pool_number = 5
    bn_axis = 3
    
    for i in range(qtd_denses):
        tensor = transition_block(tensor, 0.5, name='pool'+str(pool_number+i))
        tensor = dense_block(tensor, blocks, name='conv'+str(conv_numbers+i))
    
    tensor = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(tensor)
    tensor = tf.keras.layers.Activation('relu', name='relu')(tensor)
    return tensor

param_space = {
    "pooling": hp.choice('pooling', ['avg', 'max', 'glob_avg', 'glob_max']),
    "optmizer": hp.choice('optmizer', ['adam', 'rmsprop']),
    "loss": hp.choice('loss', ['categorical_crossentropy']),
    "unfreeze": hp.choice('unfreeze', [10, 15, 20, 30, 40]),
    "dense_blocks": hp.choice('dense_blocks', [
        {
            'qtd_denses': 0, 
            'blocks': []
        },
        {
            'qtd_denses': 1, 
            'blocks': hp.choice('blocks2', [6, 12, 16, 24, 32, 48])
        },
        {
            'qtd_denses': 2, 
            'blocks': hp.choice('blocks3', [6, 12, 16, 24, 32, 48])
        },
    ]),
    "dense": hp.choice('dense', [
        {
            'qtd_dense': 1,
            'filters': hp.choice('filters4', [[256], [512], [1024]]),
            'activation': hp.choice('activation', ['relu', 'elu']),
            'dropout': hp.choice('dropout', [0.5, 0.6, 0.7])
        },
        {
            'qtd_dense': 2,
            'filters': hp.choice('filters5', [[256, 256], [256, 512]]),
            'activation': hp.choice('activation2', ['relu', 'elu']),
            'dropout': hp.choice('dropout2', [0.5, 0.6, 0.7])
        },
        {
            'qtd_dense': 3,
            'filters': hp.choice('filters6', [[256, 256, 256], [256, 512, 1024]]),
            'activation': hp.choice('activation3', ['relu', 'elu']),
            'dropout': hp.choice('dropout3', [0.5, 0.6, 0.7])
        },
        {
            'qtd_dense': 4,
            'filters': hp.choice('filters7', [[256, 256, 256, 256], [256, 512, 1024, 2048]]),
            'activation': hp.choice('activation4', ['relu', 'elu']),
            'dropout': hp.choice('dropout4', [0.5, 0.6, 0.7])
        }
    ])
}

def hyperopt_fitness(params: dict):        
    dense_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=tf.keras.Input((355, 370, 3)))
    dense_model.layers.pop()
    dense_model.layers.pop()

    tensor = dense_block_change_choice(dense_model.layers[-3].output, params["dense_blocks"]["qtd_denses"], params["dense_blocks"]["blocks"])
    tensor = pooling_choice(tensor, params["pooling"])
    tensor = dense_choice(tensor, params["dense"]["qtd_dense"], params["dense"]["filters"], params["dense"]["activation"], params["dense"]["dropout"])
    tensor = tf.keras.layers.Dense(5, activation="softmax") (tensor)
    
    model = tf.keras.Model(inputs=dense_model.input, outputs=tensor)

    for layer in dense_model.layers[params["unfreeze"]:]:
        layer.trainable = False
    
    model.compile(optimizer=params["optmizer"], loss=params["loss"], metrics=['accuracy'])

    train_steps = 6177 / 3
    valid_steps = 1544 / 8
    # test_steps = 856 / 8

    model.fit(train_gen, epochs=30, steps_per_epoch=train_steps, validation_data=valid_gen, validation_steps=valid_steps, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

    # preds_labels = model.predict(test_gen)
    
    _, acc = model.evaluate(test_gen, verbose=2)

    results = {
        'loss': 1-acc,
        'acurracy': acc,
        'space': params,
        'status': STATUS_OK
    }

    del model
    tf.keras.backend.clear_session()
    
    save_result(results)
    return results

def save_result(resultado):
    with open('resultados_new.csv','a', newline='') as results:
        writer = csv.writer(results)      
        writer.writerow([resultado['acurracy'], resultado['loss'], resultado['space']])

def run_a_trial():
    try:
        trials = pickle.load(open("otimizacao_densenet.pkl", "rb"))
        print("Encontrei uma otimização já salva! Carregando...")
        max_evals = len(trials.trials) + 1
        print("Rodando a partir da {} iteração.".format(
        len(trials.trials)))
    except:
        trials = Trials()
        print("Começando do zero.")

    trials = Trials()
    best = fmin(hyperopt_fitness, 
                param_space,
                algo=tpe.suggest, 
                max_evals=1, 
                trials=trials)
    
    pickle.dump(trials, open("otimizacao_densenet.pkl", "wb"))

while True:
    try:
        run_a_trial()
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)