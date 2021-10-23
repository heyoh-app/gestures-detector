import torch

import coremltools as ct

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op
from coremltools.models import model as coreml


@register_torch_op
def masked_select(context, node):
    indices = context[node.inputs[1]]
    x = context[node.inputs[0]]
    ind = mb.non_zero(x=indices)

    def false_fn():
        idx = mb.slice_by_size(x=ind, begin=[0, 1], size=[-1, -1])
        res = mb.gather_nd(x=x, indices=idx)
        return res

    def true_fn():
        ind_shape = mb.shape(x=ind)
        length = mb.slice_by_index(x=ind_shape, begin=[0], end=[1], stride=[1])
        res = mb.fill(shape=length)
        return res

    shape = mb.shape(x=ind)
    eq = mb.equal(x=shape, y=[0])

    res = mb.cond(pred=eq, _true_fn=true_fn, _false_fn=false_fn, name=node.name)
    context.add(res)


def convert_to_coreml(model: torch.jit.ScriptModule, input_size: tuple) -> coreml.MLModel:
    image_scale = 1.0 / (255.0 * 0.226)
    image_bias = -0.449 / 0.226

    model = ct.convert(
        model,
        inputs=[
            ct.ImageType(
                name="input",
                shape=input_size,
                bias=[image_bias, image_bias, image_bias],
                scale=image_scale,
            )
        ],
    )

    return model
