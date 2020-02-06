from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, Concatenate, Softmax, Multiply
from keras.engine import Model
from .unet import create_convolution_block, concatenate


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def attention_unet_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, activation_name="sigmoid"):
    """
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list() #result of each level after doing summation of residual
    level_filters = list() #number of filters regards each level
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            # level 0 (up-most) of left path doesnot stride
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            # other levels of left path stride 2
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        #after 3x3x3 conv (and stride 2x2x2), pass through context modules
        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        #using result of context module and result of conv 3x3x3 to perform residual learning
        summation_layer = Add()([in_conv, context_output_layer])
        
        #position on left path before passing on "-- line" to get to right path
        level_output_layers.append(summation_layer)
        #at the end of the for loop, "current_layer" will be level 4 (bottom-most one)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        #start upsampling (from level 4)
        #the first iter: upsampling result of level 4 to scale of level 3
        # up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        # concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)

        """
        Starting attention gating
        """
        x = Conv3D(level_filters[level_number], (1,1,1))(level_output_layers[level_number]) #level 3
        g = Conv3D(level_filters[level_number], (1,1,1))(current_layer) #level 4
        g = UpSampling3D(size=2)(g)
        concat = Concatenate(axis=1)([x,g])
        relu = Activation('relu')(concat)
        psi = Conv3D(level_filters[level_number], (1,1,1))(relu) #level 4
        sigmoid = Activation('sigmoid')(psi)
        # upsampled = F.upsample(sigmoid, size=(2,2,2), mode='trilinear') #size thi bang cho cai ma no cbi concat
        concatenation_layer = Multiply()([x, sigmoid])

        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = None

    if activation_name == 'sigmoid':
        activation_block = Activation(activation_name)(output_layer)
    elif activation_name == 'softmax':
        activation_block = Softmax(axis=1)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    return model


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2