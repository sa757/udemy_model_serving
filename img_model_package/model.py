from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from img_model_package.modeling_pipeline.config import config


def img_model(
        kernel_size=(3, 3),
        pool_size=(2, 2),
        stride_size=(2, 2),
        dropout_rate=0.5,
        first_filters=32,
        second_filters=64,
        third_filters=128,
        flatten_unit=256,
        class_num=12,
        image_size=config.IMAGE_SIZE):
    # can't run heavy model due to my resource
    cnnmodel = Sequential()
    cnnmodel.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same',
                        input_shape=(image_size, image_size, 3)))
    cnnmodel.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same'))
    cnnmodel.add(MaxPooling2D(pool_size=pool_size, strides=stride_size))
    cnnmodel.add(Dropout(dropout_rate))

    cnnmodel.add(Conv2D(second_filters, (3, 3), activation='relu', padding='same'))
    cnnmodel.add(Conv2D(second_filters, (3, 3), activation='relu', padding='same'))
    cnnmodel.add(MaxPooling2D(pool_size=pool_size, strides=stride_size))
    cnnmodel.add(Dropout(dropout_rate))

    cnnmodel.add(Conv2D(third_filters, kernel_size, activation='relu', padding='same'))
    cnnmodel.add(Conv2D(third_filters, kernel_size, activation='relu', padding='same'))
    cnnmodel.add(MaxPooling2D(pool_size=pool_size, strides=stride_size))
    cnnmodel.add(Dropout(dropout_rate))

    cnnmodel.add(Flatten())
    cnnmodel.add(Dense(flatten_unit, activation="relu"))
    cnnmodel.add(Dropout(dropout_rate))
    cnnmodel.add(Dense(class_num, activation="softmax"))

    cnnmodel.compile(
        Adam(lr=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return cnnmodel


checkpoint = ModelCheckpoint(
    config.MODEL_PATH,
    monitor='acc',
    verbose=1,
    save_best_only=True,
    mode='max')

reduce_lr = ReduceLROnPlateau(
    monitor='acc',
    factor=0.5,
    patience=1,
    verbose=1,
    mode='max',
    min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

cnn_clf = KerasClassifier(build_fn=img_model,
                          batch_size=config.BATCH_SIZE,
                          validation_split=10,
                          epochs=config.EPOCHS,
                          verbose=1,  # progress bar - required for CI job
                          callbacks=callbacks_list,
                          image_size=config.IMAGE_SIZE
                          )

if __name__ == '__main__':
    model = img_model()
    model.summary()