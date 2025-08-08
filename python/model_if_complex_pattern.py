# Test complex pattern of If node
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import time
import os

def new_const_dim(val):
    return op.Constant(Type.i64, Shape([len(val)]), val)

def graph_then_branch(if_input, raw_images_1, resize_target_shape, broadcast_shape):
    raw_images_f32_1 = opset.convert(raw_images_1, Type.f32, "then_convert")
    img_trans_1 = opset.transpose(raw_images_f32_1, new_const_dim([0, 3, 1, 2]), "then_transpose")

    image_mean = op.Constant(Type.f32, Shape([1,3,1,1]), [0.1,0.1,0.1])
    image_scale = op.Constant(Type.f32, Shape([1,3,1,1]), [0.1,0.1,0.1])

    attributes = {
        "axes": [2, 3],
        "mode": "cubic",
        "antialias":True,
        "align_corners":True
    }
    img_resized_1 = opset.interpolate(img_trans_1, resize_target_shape, attributes, name="then_interpolate")
    img_resized_rnd_1 = opset.round(img_resized_1, "HALF_TO_EVEN", name="then_round")
    resized_images_f32_planar_1 = opset.clamp(img_resized_rnd_1, 0, 255, name="then_clamp")
    resized_images_m_1 = opset.subtract(resized_images_f32_planar_1, image_mean, name="then_subtract")
    resized_images_s_1 = opset.multiply(resized_images_m_1, image_scale, name="then_multiple")
    temporal_images = opset.broadcast(resized_images_s_1, broadcast_shape, name="then_broadcase")

    results = ov.opset6.result(temporal_images, "then_res")
    submodel = Model(results=[results], parameters=[if_input, raw_images_1,
                                                    resize_target_shape,
                                                    broadcast_shape], name='then_body')
    return submodel, results

def graph_else_branch(else_same_image, raw_images_1, raw_images_2, resize_target_shape):
    raw_images_f32_1 = opset.convert(raw_images_1, Type.f32, name="else_convert1")
    raw_images_f32_2 = opset.convert(raw_images_2, Type.f32, name="else_convert2")
    
    image_mean = op.Constant(Type.f32, Shape([1,3,1,1]), [0.1,0.1,0.1])
    image_scale = op.Constant(Type.f32, Shape([1,3,1,1]), [0.1,0.1,0.1])

    img_trans_1 = opset.transpose(raw_images_f32_1, new_const_dim([0, 3, 1, 2]), name="else_transpose1")
    img_trans_2 = opset.transpose(raw_images_f32_2, new_const_dim([0, 3, 1, 2]), name="else_transpose2")
    attributes = {
        "axes": [2, 3],
        "mode": "cubic",
        # "pads_begin": np.array([2, 2], dtype=dtype),
        "antialias":True,
        "align_corners":True
    }
    img_resized_1 = opset.interpolate(img_trans_1, resize_target_shape, attributes, name="else_interpolate1")
    img_resized_2 = opset.interpolate(img_trans_2, resize_target_shape, attributes, name="else_interpolate2")

    img_resized_rnd_1 = opset.round(img_resized_1, "HALF_TO_EVEN", name="else_round1")
    img_resized_rnd_2 = opset.round(img_resized_2, "HALF_TO_EVEN", name="else_round2")
    resized_images_f32_planar_1 = opset.clamp(img_resized_rnd_1, 0, 255, name="else_clamp1")
    resized_images_f32_planar_2 = opset.clamp(img_resized_rnd_2, 0, 255, name="else_clamp2")
    resized_images_m_1 = opset.subtract(resized_images_f32_planar_1, image_mean)
    resized_images_m_1 = opset.subtract(resized_images_m_1, else_same_image)
    resized_images_m_2 = opset.subtract(resized_images_f32_planar_2, image_mean)
    resized_images_s_1 = opset.multiply(resized_images_m_1, image_scale)
    resized_images_s_2 = opset.multiply(resized_images_m_2, image_scale)
    temporal_images = opset.concat([resized_images_s_1, resized_images_s_2], axis=0, name="my_concat")

    results = ov.opset6.result(temporal_images, "res")
    submodel = Model(results=[results], parameters=[else_same_image, raw_images_1,
                                                    raw_images_2,
                                                    resize_target_shape], name='else_body')
    return submodel, results

def model_if_complex():
    same_image = opset.parameter([1], Type.f32, "same_image")

    raw_images_1 = opset.parameter([-1, -1, -1, -1], Type.u8, "raw_images_1")
    raw_images_2 = opset.parameter([-1, -1, -1, -1], Type.u8, "raw_images_2")
    resize_target_shape = opset.parameter([2], Type.i64, "resize_target_shape")
    broadcast_shape = opset.parameter([4], Type.i64, "broadcast_shape")

    then_same_image = opset.parameter([1], Type.f32, "then_same_image")
    then_raw_images_1 = opset.parameter([-1, -1, -1, -1], Type.u8, name="then_inp_1")
    then_resize_target_shape = opset.parameter([2], Type.i64, name="then_inp_3")
    then_broadcast_shape = opset.parameter([4], Type.i64, name="then_inp_6")

    model_then, output_then = graph_then_branch(then_same_image, then_raw_images_1,
                                    then_resize_target_shape,
                                    then_broadcast_shape)
    
    else_same_image = opset.parameter([1], Type.f32, "else_same_image")
    else_raw_images_1 = opset.parameter([-1, -1, -1, -1], Type.u8)
    else_raw_images_2 = opset.parameter([-1, -1, -1, -1], Type.u8)
    else_resize_target_shape = opset.parameter([2], Type.i64)
    model_else, output_else = graph_else_branch(else_same_image, else_raw_images_1,
                                   else_raw_images_2,
                                   else_resize_target_shape)

    if_op = ov.opset8.if_op()
    if_op.set_then_body(model_then)
    if_op.set_else_body(model_else)

    # Note: "IF"节点，第一个参数，判断条件必须参与计算，否则可能无法被注册。无法理解其行为。 
    if_op.set_input(same_image.output(0), then_same_image, else_same_image)

    if_op.set_input(raw_images_1.output(0), None, else_raw_images_1)
    if_op.set_input(raw_images_2.output(0), None, else_raw_images_2)
    if_op.set_input(resize_target_shape.output(0), None, else_resize_target_shape)
    
    if_op.set_input(raw_images_1.output(0), then_raw_images_1, None)
    if_op.set_input(resize_target_shape.output(0), then_resize_target_shape, None)
    if_op.set_input(broadcast_shape.output(0), then_broadcast_shape, None)

    result_if = if_op.set_output(output_then, output_else)

    # add = opset.add(result_if, op.Constant(np.full((1), 0).astype(np.float32)))
    res = ov.opset6.result(result_if, "res")

    return Model(results=[res], parameters=[same_image, raw_images_1, raw_images_2, resize_target_shape, broadcast_shape], name='model_if')

def test_then_branch(device:str):
    print(f'== test_then_branch: device = {device}')
    core = Core()

    then_same_image = opset.parameter([1], Type.f32)
    then_raw_images_1 = opset.parameter([-1, -1, -1, -1], Type.u8, name="then_inp_1")
    then_resize_target_shape = opset.parameter([2], Type.i64, name="then_inp_3")
    then_broadcast_shape = opset.parameter([4], Type.i64, name="then_inp_6")

    model, outputs = graph_then_branch(then_same_image, then_raw_images_1,
                                    then_resize_target_shape,
                                    then_broadcast_shape)

    compiled_model = core.compile_model(model=model, device_name=device)
    
    same_image=np.array([1]).astype(np.float32)
    raw_images_1 = (np.random.randn(1, 128, 128, 3)*20).astype(np.uint8)
    resize_target_shape = np.array([140, 140]).astype(np.int64)
    broadcast_shape = np.array([2, 3, 140, 140]).astype(np.int64)

    infer_request = compiled_model.create_infer_request()

    print(f"== Run model_if, device={device}")
    result = infer_request.infer([same_image, raw_images_1, 
                                  resize_target_shape,
                                  broadcast_shape])[compiled_model.output(0)]
    print(f'== reuslt = {result.shape}')

def test_model_if_complex(device:str, then_branch=True):
    print(f'== test_model_if_complex device = {device}, then_branch = {then_branch}')
    core = Core()
    model = model_if_complex()
    compiled_model = core.compile_model(model=model, device_name=device)
    
    same_image=np.array([1] if then_branch else [0]).astype(np.float32)
    raw_images_1 = (np.random.randn(1, 128, 128, 3)*20).astype(np.uint8)
    raw_images_2 = (np.random.randn(1, 128, 128, 3)*20).astype(np.uint8)
    resize_target_shape = np.array([140, 140]).astype(np.int64)
    broadcast_shape = np.array([2, 3, 140, 140]).astype(np.int64)

    infer_request = compiled_model.create_infer_request()

    print(f"== Run model_if, device={device}")
    result = infer_request.infer([same_image, raw_images_1, raw_images_2, 
                                  resize_target_shape,
                                  broadcast_shape])[compiled_model.output(0)]
    print(f'== reuslt = {result.shape}')

if __name__ == "__main__":
    print(f'ov version:{ov.get_version()}')
    test_model_if_complex('GPU', then_branch=True)
    test_model_if_complex('GPU', then_branch=False)
    # test_then_branch("CPU") # pass
    