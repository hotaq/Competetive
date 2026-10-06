

import math
import numpy as np
import onnx
import spox
from spox import argument, Tensor
from spox.opset.ai.onnx import v17 as op

# don't change taskname or elase automated evaluation don't work
taskname = "task109"

"""
TASK
arc-gen task generator: task_47c1f68c.py
Task109 final report: gather-crop + reflected sentinel canvas + two-Where recolor.


ONNX SOLVER
Final chosen version
--------------------
This closes on the v11 `where_color` design rather than the later v12 split-bg/fg
attempt. v12 was slightly worse because the split foreground path introduced an
extra concat/materialization; v11 kept a cleaner two-Where color construction and
scored marginally better in the profiler.

Verified score snapshot for the chosen direction
------------------------------------------------
The v11 profile reached approximately:
    MACs   = 1058
    memory = 5898 bytes
    params = 631
    cost   = 7587
    score  = 16.066
with correctness passing on all 266 examples in the local evaluation flow.

Core trick
----------
1. Detect the object/line scale from the first background row:
   - slice bg row [0:14]
   - compute adjacent edge x[i] - x[i+1]
   - multiply by index and ReduceMax to get the useful transition position
   - derive cropsize = (S_less_one + 1) // 2.

2. Avoid dynamic s-by-s slicing:
   - slice a fixed 6x6 background mask from input
   - build fixed 36 gather indices from cropsize
   - gather the true cropsize-by-cropsize object mask into the bottom-right of a
     fixed 6x6 canvas
   - use dummy index/value for invalid cells.

3. Reflect by fixed Gather, not ReverseSequence:
   - gather columns [5,4,3,2,1,0]
   - concat left/right
   - gather rows [5,4,3,2,1,0]
   - concat top/bottom
   This creates a fixed 12x12 reflected canvas without dynamic shapes.

4. Preserve the 255 sentinel semantics:
   - 0 means real background
   - 1 means object mask / line-colored object
   - 255 means transparent padding/sentinel and must become all-zero before the
     final dynamic pad.
   Earlier attempts that removed 255 or moved Equal after final Pad were wrong or
   much more expensive.

5. Recolor using the line color, not original object colors:
   - sample input[:, :, 0, S] as the line color
   - use two Where operations to map the 12x12 id/mask canvas into one-hot color:
       * real background -> channel 0
       * object -> sampled line color
       * 255 sentinel -> all-zero

6. Final dynamic Pad:
   - crop away the top/left offset using negative pads
   - pad bottom/right to 30x30
   - patch the dynamic Pad output shape for the Kaggle memory profiler.

Why stop here
-------------
The easy gains have been taken:
- replaced dynamic crop/pad with fixed gather canvas
- replaced original-color extraction with background-channel mask
- switched large int64 mask/index work where possible to uint8/int32
- cached constants
- avoided the expensive 10-way Equal(depth) path from earlier versions.

The remaining score bottleneck is mostly unavoidable materialization around the
12x12 color construction and the fixed gather-index logic. A later v12 split-bg /
foreground experiment was slightly worse, so v11 is the better closing point.
Further gains would likely need a new structural trick, not another small cleanup.
"""

# don't change existing helper functions. you can add new ones
#helper ---------------------------------------------------------------
_CONST_CACHE = {}


def reset_const_cache():
    _CONST_CACHE.clear()


def _make_const_array(value, dtype=np.int64, shape=None):
    if not isinstance(value, np.ndarray):
        if shape is not None:
            arr = np.full(shape, value, dtype=dtype)
        else:
            arr = np.array(value, dtype=dtype)
    else:
        arr = np.asarray(value, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
    return np.ascontiguousarray(arr)


def op_constant(value, dtype=np.int64, shape=None):
    arr = _make_const_array(value, dtype=dtype, shape=shape)
    key = (arr.dtype.str, tuple(arr.shape), arr.tobytes())
    cached = _CONST_CACHE.get(key)
    if cached is not None:
        return cached
    out = op.constant(value=arr)
    _CONST_CACHE[key] = out
    return out

def op_slice(x, starts, ends, axes=None, steps=None):
    if isinstance(starts, list):
        starts=op_constant(starts, np.int64)
    if isinstance(ends, list):
        ends=op_constant(ends, np.int64)
    if isinstance(axes, list):
        axes=op_constant(axes, np.int64)
    if isinstance(steps, list):
        steps=op_constant(steps, np.int64)
    return op.slice(x, starts, ends, axes, steps)









#####################################################################
def debug_names(model):
    print("\n=== INPUTS ===")
    for x in model.graph.input:
        print(x.name)

    print("\n=== OUTPUTS ===")
    for x in model.graph.output:
        print(x.name)

    print("\n=== INITIALIZERS ===")
    for x in model.graph.initializer:
        print(x.name)

    print("\n=== NODE OUTPUTS ===")
    for node in model.graph.node:
        for out in node.output:
            print(out)

def debug_shapes(model):
    model = onnx.shape_inference.infer_shapes(
        model, strict_mode=False)  # this must be called before patch_shapes,debug_shapes

    print("\n=== VALUE INFO ===")
    for vi in model.graph.value_info:
        name = vi.name
        t = vi.type.tensor_type
        if not t.HasField("shape"):
            continue

        shape = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                shape.append(f"sym({d.dim_param})")
            else:
                shape.append("?")
        print(f"{name:30s} -> {shape} ({t.elem_type})")

    print("\n=== NODE OUTPUT MAP ===")
    for node in model.graph.node:
        for out in node.output:
            print(f"{out:30s} <- {node.op_type}")

def patch_shapes(model, changes={}):
    names = list( changes.keys())
    shapes = list( changes.values())
    model = onnx.shape_inference.infer_shapes(
        model, strict_mode=False)  # this must be called before patch_shapes,debug_shapes

    """
    Patch tensor shapes in graph.value_info / graph.output / graph.input.

    names:  list of tensor names
    shapes: list of shapes, e.g. [[1,10,30,30], [1,10,30,30]]
    """
    assert len(names) == len(shapes)

    def set_shape(value_info, shape):
        t = value_info.type.tensor_type
        # Clear old shape
        t.ClearField("shape")

        for s in shape:
            dim = t.shape.dim.add()
            if isinstance(s, int):
                dim.dim_value = s
            else:
                dim.dim_param = str(s)

    all_infos = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )

    existing = {vi.name: vi for vi in all_infos}

    for name, shape in zip(names, shapes):
        if name in existing:
            set_shape(existing[name], shape)
        else:
            # default FLOAT if tensor is not already in value_info
            vi = onnx.helper.make_tensor_value_info(
                name,
                onnx.TensorProto.FLOAT,
                shape,
            )
            model.graph.value_info.append(vi)
    return model

##########################################################################
def make_onnx():
    reset_const_cache()
    SHAPE = (1, 10, 30, 30) #padH, padW
    input = argument(Tensor(np.float32, SHAPE))

    #================== start of code ================================
    # detect orginal H,W = S,S
    # largest is 13x13

    bg = op_slice(input, [0, 0, 0, 0], [1, 1, 1, 14], [0, 1, 2, 3], [1, 1, 1, 1])
    bg = op.cast(bg, to=np.int8)
    bg = op.reshape(bg, shape=op_constant([14])) #todo: remove this
    x0 = op_slice(bg, [0], [13])
    x1 = op_slice(bg, [1], [14])
    edge = op.sub(x0, x1)

    idx = op_constant(np.arange(13), dtype=np.int8)
    edge = op.mul(edge,idx)
    S_less_one = op.reduce_max(edge) #hck:verify correct
    cropsize = op.div(op.add(S_less_one,
        op_constant(1, dtype=np.int8)),op_constant(2, dtype=np.int8))
    #cropsize is (S_less_one+1)//2


    #line color --------------------------------------------------
    S = cropsize#op.add(S_less_one, op_constant(1, dtype=np.int8))
    S_1d = op.cast(op.reshape(S, op_constant([1], dtype=np.int64)), to=np.int64)
    S1_1d = op.add(S_1d, op_constant([1], dtype=np.int64))

    line_starts = op.concat(
        [
            op_constant([0, 0, 0], dtype=np.int64),
            S_1d,
        ],
        axis=0,
    )

    line_ends = op.concat(
        [
            op_constant([1, 10, 1], dtype=np.int64),
            S1_1d,
        ],
        axis=0,
    )

    linecolor = op.slice(
        input,
        starts=line_starts,
        ends=line_ends,
        axes=op_constant([0, 1, 2, 3], dtype=np.int64),
        steps=op_constant([1, 1, 1, 1], dtype=np.int64),
    )  # [1,10,1,1] one-hot float
    linecolor_u8 = op.cast(linecolor, to=np.uint8)

    # fixed 6x6 color crop from input, then gather-shift the true cropsize x cropsize
    # object to the bottom-right of a 6x6 canvas. Invalid/padding cells use 255.


    #todo: fixed this!
    #crop6_oh = op_slice(input, [0, 0, 0, 0], [1, 10, 6, 6], [0, 1, 2, 3], [1, 1, 1, 1])
    #crop6_color = op.arg_max(crop6_oh, axis=1, keepdims=1)       # [1,1,6,6], int64


    bg6 = op_slice(input, [0, 0, 0, 0], [1, 1, 6, 6], [0, 1, 2, 3], [1, 1, 1, 1])

    crop6_mask = op.less(bg6,
                         op_constant(0.5, dtype=np.float32, shape=[1, 1, 1, 1])
                         )
    # keep this path uint8 instead of int64: mask(0/1) * dynamic line color id
    # This removes a large [1,1,6,6] int64 cast/mul buffer from the profile.
    # Keep a binary object mask through gather/reflect: 0=background, 1=object, 255=sentinel.
    # The final color is applied later by Where using linecolor_u8.
    crop6_color = op.cast(crop6_mask, to=np.uint8)
    crop6_flat = op.reshape(crop6_color, op_constant([36], dtype=np.int64))
    crop6_flat37 = op.concat(
        [crop6_flat, op_constant(255, dtype=np.uint8, shape=[1])],
        axis=0,
    )




    # Output canvas coords in 6x6. We want destination bottom-right:
    #   src_y = dst_y - (6-cropsize), src_x = dst_x - (6-cropsize)
    # invalid dst cells gather dummy index 36 -> 255.
    # Keep gather-index arithmetic in int32. Gather accepts int32 indices,
    # and this halves the 36-vector index buffers compared with int64.
    cropsize_i32 = op.cast(cropsize, to=np.int32)
    off = op.sub(op_constant(6, dtype=np.int32), cropsize_i32)
    yy6 = op_constant(np.repeat(np.arange(6), 6), dtype=np.int32)
    xx6 = op_constant(np.tile(np.arange(6), 6), dtype=np.int32)
    sy = op.sub(yy6, off)
    sx = op.sub(xx6, off)
    valid_y6 = op.greater_or_equal(sy, op_constant(0, dtype=np.int32))
    valid_x6 = op.greater_or_equal(sx, op_constant(0, dtype=np.int32))
    valid6 = op.and_(valid_y6, valid_x6)
    src_index = op.add(op.mul(sy, op_constant(6, dtype=np.int32)), sx)
    src_index_safe = op.where(
        valid6,
        src_index,
        op_constant(36, dtype=np.int32, shape=[36]),
    )

    canvas_flat = op.gather(crop6_flat37, indices=src_index_safe, axis=0)
    canvas = op.reshape(canvas_flat, op_constant([6, 6], dtype=np.int64))  # uint8, 255 outside object

    # Reflect the bottom-right-aligned 6x6 color canvas to 12x12.
    reverse_idx = op_constant([5, 4, 3, 2, 1, 0], dtype=np.int64)
    right  = op.gather(canvas, indices=reverse_idx, axis=1)
    upper  = op.concat([canvas, right], axis=1)
    bottom = op.gather(upper, indices=reverse_idx, axis=0)
    reflected = op.concat([upper, bottom], axis=0)

    # Mask right/bottom outside [0 : 6+cropsize) to 255. Top/left are already
    # represented by 255 from the bottom-right gather shift and will be removed by
    # the final negative pad/crop.
    keep = op.add(op_constant(6, dtype=np.int8), op.cast(cropsize, to=np.int8))
   # keep = op.reshape(keep,shape=op_constant([1,1]))
    yy   = op_constant(np.repeat(np.arange(12), 12), dtype=np.int8)
    xx   = op_constant(np.tile(np.arange(12), 12), dtype=np.int8)
    valid_y = op.less(yy, keep)
    valid_x = op.less(xx, keep)
    valid = op.and_(valid_y, valid_x)  # [144]
    flat  = op.reshape(reflected, op_constant([144], dtype=np.int64))

    reflected_padded = op.where(
        valid,
        flat,
        op_constant(255, dtype=np.uint8, shape=[144]),
    )
    reflected_padded = op.reshape(
        reflected_padded,
        op_constant([12, 12], dtype=np.int64),
    )

    # Convert 0/1/255 id map to one-hot-like uint8 without 10-way Equal:
    #   0   -> background one-hot
    #   1   -> sampled linecolor one-hot
    #   255 -> all-zero transparent sentinel
    out_id = op.reshape(reflected_padded, op_constant([1, 1, 12, 12]))
    is_bg = op.equal(out_id, op_constant(0, dtype=np.uint8, shape=[1, 1, 1, 1]))
    is_valid = op.less(out_id, op_constant(255, dtype=np.uint8, shape=[1, 1, 1, 1]))
    bg_color = op_constant(np.array([1,0,0,0,0,0,0,0,0,0], dtype=np.uint8), dtype=np.uint8, shape=(1, 10, 1, 1))
    zero_color = op_constant(0, dtype=np.uint8, shape=(1, 10, 1, 1))
    out_valid = op.where(is_bg, bg_color, linecolor_u8)
    out = op.where(is_valid, out_valid, zero_color)

    #final padding to 1x10x30x30
    cropsize = op.cast(cropsize, to=np.int8)
    left0 = op.sub(op_constant([6], np.int8), cropsize)
    left = op.sub(op_constant([0], np.int8), left0)
    right = op.sub(
        op_constant([30], np.int8),
        op.add(op_constant([6], np.int8), cropsize),
    )

    pads = op.cast(#todo: can index be int32?
        op.concat(
            [
                op_constant([0, 0], dtype=np.int8),
                left,
                left,
                op_constant([0, 0], dtype=np.int8),
                right,
                right,
            ],
            axis=0,
        ),
        to=np.int64,
    )

    output = op.pad(
        out,
        pads=pads,
        constant_value=op_constant(0, dtype=np.uint8),
    )

    ##############################################################

    model = spox.build(inputs={"input": input}, outputs={
        'output': output, #op.cast(input, to=np.uint8),
        #'out': output,
        # 'bg': bg,
        # 'edge': edge,
        # 'S_less_one': S_less_one,
        # 'cropsize': cropsize,
        # 'reflected': reflected,
        # 'reflected_padded': reflected_padded,
    })

    #debug_names(model)
    #model = onnx.shape_inference.infer_shapes(model, strict_mode=False) #this must be called before patch_shapes,debug_shapes
    #debug_shapes(model)


    model = patch_shapes(model,{
        'Pad_0_output': [1,10,30,30],
        #'Slice_3_output': [1,1,1,1],
        'Slice_4_output': [1,10,1,1],
    })
    return model

######################################################################################
# Keep the same template/evaluation style as task111_v02.
from onnxsim import simplify
import onnxruntime
from my_evalution import *


if __name__ == "__main__":
    onnx_file = f"{taskname}.onnx"

    model = make_onnx()
    model, ok = simplify(model)
    # do not re-apply patch after simplify,
    onnx.checker.check_model(model)
    onnx.save(model, onnx_file)
    print(f"{onnx_file}.onnx, simplify ok={ok}")

    # --- check score profile -----
    if 1:
        macs, params = print_onnx_tool_profile(onnx_file)
        memory = print_kaggle_memory_proile(onnx_file)

        print()

        cost = macs + memory + params

        score = max(1.0, 25.0 - math.log(max(1.0, cost)))
        print(
            f"It appears to require {macs} MACs + {memory} bytes + {params} params, "
            f"yielding cost = {cost} and score = {score:.3f}"
        )
        print('check:', (score_network(onnx_file)))
        print()

    # --- check correctness -----
    session = onnxruntime.InferenceSession(model.SerializeToString())

    examples = load_examples(taskname)
    examples = clean_examples(examples, taskname)
    examples = examples["arc-gen"] + examples["train"] + examples["test"]
    #examples = examples[:10]

    debug_key=[]
    for out in model.graph.output:
        if out.name!='output':
            debug_key.append(out.name)

    wrong = 0
    for i in range(len(examples)):
        print("i", i, "--------------------------------")
        e = examples[i]

        inp = np.array(e["input"])
        inp = convert_color_to_onehot(inp)
        truth = np.array(e["output"])
        truth = convert_color_to_onehot(truth)

        (predict,) = session.run(["output"], {"input": inp})
        print("predict", predict.shape)
        #print(predict[0,0])
        #continue
        if 0:
            predict = session.run(debug_key, {'input': inp})
            for k, p in zip(debug_key, predict):
                print(k, p.shape)
                print(p)
            # predict = predict.astype(np.float32)
            print()

            continue


        predict = predict.astype(np.float32)
        wng = 1 - np.array_equal(predict, truth)
        wrong += wng

        if (i < 2) or (wng):
            print("input")
            print_grid_color(convert_onehot_to_color(inp))
            print("predict")
            print_grid_color(convert_onehot_to_color(predict))
            print("truth")
            print_grid_color(convert_onehot_to_color(truth))

    print("DONE !!!====================")
    print("len(examples)", len(examples))
    print("wrong", wrong)


''''
numpy version:  2.4.4
onnx version:  1.21.0
onnxruntime version:  1.24.4
onnx_tool version:  1.0.1
NEUROGOLF_FILE_VERSION:  28-apr.01

task109.onnx.onnx, simplify ok=True
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
Name              Type              Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
----------------  --------------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
Add_3             Add                          1  0.09%             16  0.53%              1  0.17%       1          1
Sub_4             Sub                          1  0.09%              9  0.30%              1  0.17%       1          1
Sub_5             Sub                          1  0.09%              9  0.30%              1  0.17%       1          1
Less_2            Less                        36  3.40%             40  1.32%              1  0.17%       1x1x6x6    1x1x6x6
Add_0             Add                          1  0.09%              2  0.07%              1  0.17%       1          1
Less_3            Less                       144  13.61%           145  4.80%              1  0.17%       1x1x12x12  1x1x12x12
Less_0            Less                       144  13.61%           288  9.54%            144  24.66%      144        144
Less_1            Less                       144  13.61%           288  9.54%            144  24.66%      144        144
And_1             And                         36  3.40%             36  1.19%              0  0.00%       36         36
ReduceMax_0       ReduceMax                   13  1.23%              1  0.03%              0  0.00%       13         1
Sub_2             Sub                         36  3.40%            288  9.54%             36  6.16%       36         36
Mul_1             Mul                         36  3.40%            148  4.90%              1  0.17%       36         36
Mul_0             Mul                         13  1.23%             26  0.86%             13  2.23%       13         13
Equal_0           Equal                      144  13.61%           145  4.80%              1  0.17%       1x1x12x12  1x1x12x12
Add_1             Add                          1  0.09%              8  0.26%              0  0.00%       1          1
Where_3           Where                        0  0.00%            154  5.10%             10  1.71%       1x1x12x12  1x1x12x12
GreaterOrEqual_1  GreaterOrEqual              36  3.40%             40  1.32%              1  0.17%       36         36
Where_0           Where                        0  0.00%            288  9.54%             36  6.16%       36         36
Where_1           Where                        0  0.00%            288  9.54%            144  24.66%      144        144
Add_2             Add                         36  3.40%            144  4.77%              0  0.00%       36         36
Where_2           Where                        0  0.00%            154  5.10%             10  1.71%       1x1x12x12  1x1x12x12
Sub_1             Sub                          1  0.09%              4  0.13%              0  0.00%       1          1
Sub_3             Sub                         36  3.40%            288  9.54%             36  6.16%       36         36
Sub_0             Sub                         13  1.23%             13  0.43%              0  0.00%       13         13
Div_0             Div                          4  0.38%              9  0.30%              1  0.17%       1          1
And_0             And                        144  13.61%           144  4.77%              0  0.00%       144        144
GreaterOrEqual_0  GreaterOrEqual              36  3.40%             36  1.19%              0  0.00%       36         36
Sub_6             Sub                          1  0.09%              9  0.30%              1  0.17%       1          1
Total             _                        1,058  100%           3,020  100%             584  100%        _          _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=      56    0.95% shape=[1, 1, 1, 14]
Cast_0_output                  int8       mem=      14    0.24% shape=[1, 1, 1, 14]
Reshape_0_reshaped             int8       mem=      14    0.24% shape=[14]
Slice_1_output                 int8       mem=      13    0.22% shape=[13]
Slice_2_output                 int8       mem=      13    0.22% shape=[13]
Sub_0_C                        int8       mem=      13    0.22% shape=[13]
Mul_0_C                        int8       mem=      13    0.22% shape=[13]
ReduceMax_0_reduced            int8       mem=       1    0.02% shape=[1]
Add_0_C                        int8       mem=       1    0.02% shape=[1]
Div_0_C                        int8       mem=       1    0.02% shape=[1]
Add_1_C                        int8       mem=       1    0.02% shape=[1]
Less_0_C                       bool       mem=     144    2.44% shape=[144]
Less_1_C                       bool       mem=     144    2.44% shape=[144]
And_0_C                        bool       mem=     144    2.44% shape=[144]
Slice_3_output                 float32    mem=     144    2.44% shape=[1, 1, 6, 6]
Less_2_C                       bool       mem=      36    0.61% shape=[1, 1, 6, 6]
Cast_2_output                  uint8      mem=      36    0.61% shape=[1, 1, 6, 6]
Reshape_1_reshaped             uint8      mem=      36    0.61% shape=[36]
Concat_0_concat_result         uint8      mem=      37    0.63% shape=[37]
Cast_3_output                  int32      mem=       4    0.07% shape=[1]
Sub_1_C                        int32      mem=       4    0.07% shape=[1]
Sub_2_C                        int32      mem=     144    2.44% shape=[36]
GreaterOrEqual_0_C             bool       mem=      36    0.61% shape=[36]
Sub_3_C                        int32      mem=     144    2.44% shape=[36]
GreaterOrEqual_1_C             bool       mem=      36    0.61% shape=[36]
And_1_C                        bool       mem=      36    0.61% shape=[36]
Mul_1_C                        int32      mem=     144    2.44% shape=[36]
Add_2_C                        int32      mem=     144    2.44% shape=[36]
Where_0_output                 int32      mem=     144    2.44% shape=[36]
Gather_0_output                uint8      mem=      36    0.61% shape=[36]
Reshape_2_reshaped             uint8      mem=      36    0.61% shape=[6, 6]
Gather_1_output                uint8      mem=      36    0.61% shape=[6, 6]
Concat_1_concat_result         uint8      mem=      72    1.22% shape=[6, 12]
Gather_2_output                uint8      mem=      72    1.22% shape=[6, 12]
Concat_2_concat_result         uint8      mem=     144    2.44% shape=[12, 12]
Reshape_3_reshaped             uint8      mem=     144    2.44% shape=[144]
Where_1_output                 uint8      mem=     144    2.44% shape=[144]
Reshape_5_reshaped             uint8      mem=     144    2.44% shape=[1, 1, 12, 12]
Less_3_C                       bool       mem=     144    2.44% shape=[1, 1, 12, 12]
Equal_0_C                      bool       mem=     144    2.44% shape=[1, 1, 12, 12]
Cast_4_output                  int64      mem=       8    0.14% shape=[1]
Concat_3_concat_result         int64      mem=      32    0.54% shape=[4]
Add_3_C                        int64      mem=       8    0.14% shape=[1]
Concat_4_concat_result         int64      mem=      32    0.54% shape=[4]
Slice_4_output                 float32    mem=      40    0.68% shape=[1, 10, 1, 1]
Cast_5_output                  uint8      mem=      10    0.17% shape=[1, 10, 1, 1]
Where_2_output                 uint8      mem=    1440   24.42% shape=[1, 10, 12, 12]
Where_3_output                 uint8      mem=    1440   24.42% shape=[1, 10, 12, 12]
Sub_4_C                        int8       mem=       1    0.02% shape=[1]
Sub_5_C                        int8       mem=       1    0.02% shape=[1]
Sub_6_C                        int8       mem=       1    0.02% shape=[1]
Concat_5_concat_result         int8       mem=       8    0.14% shape=[8]
Cast_7_output                  int64      mem=      64    1.09% shape=[8]
output                         uint8      mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=    5898 100.00%

It appears to require 1058 MACs + 5898 bytes + 631 params, yielding cost = 7587 and score = 16.066
check: (1058, 5898, 631)
len(examples) 266
wrong 0
'''