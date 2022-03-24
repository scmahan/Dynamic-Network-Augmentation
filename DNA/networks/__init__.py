from .wideresnet import WideResNet


def get_model(model_name='wresnet40_2', num_class=10, policy_shape=None):
    name = model_name

    if name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class, policy_shape=policy_shape)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class, policy_shape=policy_shape)

    return model

