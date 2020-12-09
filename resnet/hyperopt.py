import tensorflow as tf

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import hp

from residual_block import stack_block

import csv, pickle, traceback

from sklearn.metrics import accuracy_score

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

def block_change_choice(tensor, qtd_stacks, filters, layers):
    conv_numbers = 6
    for i in range(qtd_stacks):
        tensor = stack_block(tensor, filters[i], layers, name="conv{}".format(conv_numbers+i))
    
    return tensor  

param_space = {
    "pooling": hp.choice('pooling', ['avg', 'max', 'glob_avg', 'glob_max']),
    "optmizer": hp.choice('optmizer', ['adam', 'rmsprop']),
    "loss": hp.choice('loss', ['categorical_crossentropy']),
    "unfreeze": hp.choice('unfreeze', [0, 10, 15, 20, 30, 40]),
    "stacks": hp.choice('stacks', [
        {
            'qtd_stacks': 0, 
            'filters': [],
            'layers': []
        },
        {
            'qtd_stacks': 1, 
            'filters': hp.choice('filters2', [[1024]]),
            'layers': hp.choice('layers2', [3, 4, 6, 8, 23, 36])
        },
        {
            'qtd_stacks': 2, 
            'filters': hp.choice('filters3', [[1024, 1024], [1024, 2048]]),
            'layers': hp.choice('layers3', [3, 4, 6, 8, 23, 36])
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
    res_model = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_tensor=tf.keras.Input((355, 370, 3)))

    tensor = block_change_choice(res_model.output, params["stacks"]["qtd_stacks"], params["stacks"]["filters"], params["stacks"]["layers"])
    tensor = pooling_choice(tensor, params["pooling"])
    tensor = dense_choice(tensor, params["dense"]["qtd_dense"], params["dense"]["filters"], params["dense"]["activation"], params["dense"]["dropout"])
    tensor = tf.keras.layers.Dense(5, activation="softmax") (tensor)

    model = tf.keras.Model(inputs=res_model.input, outputs=tensor)

    for layer in res_model.layers[params["unfreeze"]:]:
        layer.trainable = False
    
    model.compile(optimizer=params["optmizer"], loss=params["loss"], metrics=['accuracy'])

    train_steps = 6177 / 3
    valid_steps = 1544 / 8
    # test_steps = 856 / 8

    model.fit(train_gen, epochs=30, steps_per_epoch=train_steps, validation_data=valid_gen, validation_steps=valid_steps, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
    
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
    with open('resultados_resnet.csv','a', newline='') as results:
        writer = csv.writer(results)      
        writer.writerow([resultado['acurracy'], resultado['loss'], resultado['space']])

def run_a_trial():
    try:
        trials = pickle.load(open("otimizacao_resnet.pkl", "rb"))
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
    
    pickle.dump(trials, open("otimizacao_resnet.pkl", "wb"))

while True:
    try:
        run_a_trial()
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)