from keras import backend as K
from keras import activations
from keras.layers import Input, Dense
from keras.models import Model
from keras.engine.topology import Layer
from keras import regularizers
import numpy as np

# Activation layer
class iAct(Layer):
    def __init__(self, activation, **kwargs):
        self.activation = activations.get(activation)
        super(iAct, self).__init__(**kwargs)

    def call(self, x):
        center, radius = x
        tmp_c = (self.activation(center - radius) + self.activation(center + radius)) / 2
        tmp_r = (self.activation(center + radius) - self.activation(center - radius)) / 2
        return [tmp_c, tmp_r]

    def compute_output_shape(self, input_shape):
        return input_shape

# Loss layer, no training required, only for making a custom loss function
class iLoss(Layer):
    def __init__(self, beta, **kwargs):
        self.beta = beta
        super(iLoss, self).__init__(**kwargs)

    def loss(self, y_true, y_pred):
        error_c = K.square(y_true[0] - y_pred[0])
        error_r = K.square(y_true[1] - y_pred[1])
        return K.mean(K.sum(self.beta * error_c + (1 - self.beta) * error_r, axis=-1))

def get_model(input_dim, output_dim, num_units, activation, beta=0.5, num_hidden_layers=1):
    center_x = Input((input_dim,), name='center_input')
    radius_x = Input((input_dim,), name='radius_input')

    for i in range(num_hidden_layers):
        c = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal', 
                kernel_regularizer=regularizers.l2(0.001))(center_x)
        r = Dense(num_units[i], use_bias=False, kernel_initializer='he_normal', 
                kernel_regularizer=regularizers.l2(0.001))(radius_x)
        c, r = iAct(activation[i])([c, r])

    c = Dense(output_dim, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(c)
    r = Dense(output_dim, use_bias=False, kernel_initializer='he_normal')(r)
    loss_layer = iLoss(beta)

    model = Model(inputs=[center_x, radius_x], outputs=[c, r])
    model.compile(loss=loss_layer.loss, optimizer='adam', metrics=['accuracy'])
    return model