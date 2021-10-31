from . import models


def build_model(args):
    model = models.__dict__[args.model](num_classes=2)
    return model
