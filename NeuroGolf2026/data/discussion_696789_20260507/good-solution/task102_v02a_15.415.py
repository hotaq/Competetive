import math
import numpy as np
import onnx
from onnxsim import simplify
import onnxruntime
from my_local_helper import *

import spox
from spox import argument, Tensor
from spox.opset.ai.onnx import v17 as op

# don't change taskname or elase automated evaluation don't work
taskname = "task102"

# don't add doctsting

"""
TASK
arc-gen task generator: task_44d8ac46.py
Solution Summary (NeuroGolf Task 102)


ONNX SOLVER
Key idea:
Detect square holes using minimal local corner primitives, then render interiors via
cheap padding-based stamping.

Detection:
- Use only gray channel (channel 5); all non-gray implicitly treated as black.
- Define four 2×2 corner primitives (P2–P5):
    P2 = [[1,1],[1,0]]
    P3 = [[1,1],[0,1]]
    P4 = [[1,0],[1,1]]
    P5 = [[0,1],[1,1]]
- For hole size h ∈ {1,2,3,4}, detect valid square by checking:
    P2(y,x), P3(y,x+h), P4(y+h,x), P5(y+h,x+h)
- Interior is treated as "don't care" due to generator guarantees.

Rendering:
- For each valid square, stamp the h×h interior by shifting the valid mask.
- Implemented via multiple Pad operations (no convolution / pooling).
- Combine all contributions using Max (logical OR in uint8).

Output construction:
- Avoid full 10-channel reconstruction.
- Build only required channels:
    channel 0 (black) = 1 - (gray + red)
    channel 2 (red)   = red mask
    channel 5 (gray)  = original gray
    others            = zero
- Concatenate and pad to 30×30.

Performance:
- Correctness: 267 / 267 (0 wrong)
- MACs:     3008
- Memory:   11010 bytes
- Params:   521
- Score:    15.415

Notes:
- Corner primitive sharing avoids expensive border scans.
- Pad-based rendering is memory-heavy but MAC-efficient; better than Conv/MaxPool here.
- Attempts with ConvTranspose / MaxPool increased cost due to MAC or index overhead.
- This version represents a good trade-off between simplicity, correctness, and score.

Possible future improvements:
- Reduce number of Pad operations (currently ~30) via grouping or more compact stamping.
- Further shrink intermediate tensors or reuse buffers to lower memory.
"""

# don't change existing helper functions. you can add new ones
############ start of helper #####################################
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

########################################################################3#

def make_onnx():

    reset_const_cache()
    input = argument(Tensor(np.float32, (1,10,30,30))) #padH, padW

    #========== start of your code here ============================
    # task120_v01_corner_primitives
    # Fixed 12x12 task.  Gray frames contain black holes; square-hole interiors become red.
    # Primitive plan:
    #   P2 = [[1,1],[1,0]]  top-left hole corner
    #   P3 = [[1,1],[0,1]]  top-right hole corner
    #   P4 = [[1,0],[1,1]]  bottom-left hole corner
    #   P5 = [[0,1],[1,1]]  bottom-right hole corner
    # A hole of size h=1..4 is detected by the four corner primitives at offsets
    # (0,0), (0,h), (h,0), (h,h).  Interior pixels between them are don't-care
    # under the generator distribution.

    def band_and4(a, b, c, d):
        return op.and_(op.and_(a, b), op.and_(c, d))

    def band_all(xs):
        out = xs[0]
        for z in xs[1:]:
            out = op.or_(out, z)
        return out

    def crop12_ch(ch):
        return op_slice(input, [0, ch, 0, 0], [1, ch + 1, 12, 12], [0, 1, 2, 3])

    def s12(x, y0, x0, h, w):
        return op_slice(x, [0, 0, y0, x0], [1, 1, y0 + h, x0 + w], [0, 1, 2, 3])

    def pad_to_12_u8(x_bool, top, left, h, w):
        x_u8 = op.cast(x_bool, to=np.uint8)
        bottom = 12 - top - h
        right = 12 - left - w
        return op.pad(
            x_u8,
            pads=op_constant([0, 0, top, left, 0, 0, bottom, right], np.int64),
        )

    gray  = op.cast(crop12_ch(5), to=np.bool_)

    g00 = s12(gray, 0, 0, 11, 11)   #slices
    g01 = s12(gray, 0, 1, 11, 11)
    g10 = s12(gray, 1, 0, 11, 11)
    g11 = s12(gray, 1, 1, 11, 11)

    p2 = band_and4(g00, g01, g10, op.not_(g11))  # 11 / 10
    p3 = band_and4(g00, g01, op.not_(g10), g11)  # 11 / 01
    p4 = band_and4(g00, op.not_(g01), g10, g11)  # 10 / 11
    p5 = band_and4(op.not_(g00), g01, g10, g11)  # 01 / 11


    red_parts = []

    for hole in [1, 2, 3, 4]:
        vh = 11 - hole
        valid = band_and4(
            s12(p2, 0,    0,    vh, vh),
            s12(p3, 0,    hole, vh, vh),
            s12(p4, hole, 0,    vh, vh),
            s12(p5, hole, hole, vh, vh),
        )

        # Render the h x h interior.  For outer top-left (y,x), interior starts at (y+1,x+1).
        for dy in range(hole):
            for dx in range(hole):
                red_parts.append(pad_to_12_u8(valid, 1 + dy, 1 + dx, vh, vh))



    #canvas = op.cast(gray, to=np.uint8)
    #canvas = op.mul(canvas, op_constant(5, np.uint8))
    #canvas = op.where(red12, op_constant(2, np.uint8), canvas)
    #depth  = op_constant(np.arange(6), np.uint8, [1, 6, 1, 1])
    #out12 = op.equal(depth, canvas)


    red12 = op.max(red_parts)
    gray12 =op.cast(gray, to=np.uint8)

    zero12 = op_constant(0, np.uint8, [1,1,12,12])
    out12  = op.concat([
        op.sub( op_constant(1, np.uint8), op.add(gray12,red12)),
        zero12,
        red12,
        zero12,
        zero12,
        gray12,
    ], axis=1)


    #========== end of your code here ============================
    output = op.pad(
        out12,
        pads=op_constant([0, 0, 0, 0, 0, 4, 18, 18], np.int64),
    )

    model = spox.build(inputs={"input": input}, outputs={
        "output": output
        # for debug
        #"var1": var1
    })
    #debug_names(model)
    #debug_shapes(model)
    onnx.checker.check_model(model)
    return model
#####################################################################################

#don't modify anything from this point onwards !!!!

if __name__ == "__main__":
    onnx_file = f"{taskname}.onnx"

    model = make_onnx()
    model, ok = simplify(model)
    onnx.checker.check_model(model)
    onnx.save(model, onnx_file)
    print(f"{onnx_file}.onnx, simplify ok={ok}")

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
    to_show = []
    for i in range(len(examples)):
        #print("i", i, "--------------------------------")
        e = examples[i]
        input = np.array(e["input"])
        input = convert_color_to_onehot(input)
        truth = np.array(e["output"])
        truth = convert_color_to_onehot(truth)

        if 0: #debug
            predict = session.run(debug_key, {'input': input})
            for k, p in zip(debug_key, predict):
                print(k, p.shape)
                print(p)
            # predict = predict.astype(np.float32)
            print()
            continue

        
        (predict,) = session.run(["output"], {"input": input})
        #print("predict", predict.shape)
        predict = predict.astype(np.float32)
        wng = 1 - np.array_equal(predict, truth)
        wrong += wng

        if (i < 5) or (wng):
            to_show.append((i, input, predict, truth))
            # print("input")
            # print_grid_color(convert_onehot_to_color(input))
            # print("predict")
            # print_grid_color(convert_onehot_to_color(predict))
            # print("truth")
            # print_grid_color(convert_onehot_to_color(truth))

    print("DONE !!!====================")
    print("len(examples)", len(examples))
    print("wrong", wrong)
    print()

    #- last check ------------
    if 1:
        for (i, input, predict, truth) in to_show:
            print("input")
            print_grid_color(convert_onehot_to_color(input))
            print("predict")
            print_grid_color(convert_onehot_to_color(predict))
            print("truth")
            print_grid_color(convert_onehot_to_color(truth))
        print("DONE !!!====================")
        print("len(examples)", len(examples))
        print("wrong", wrong)
        print(
            f"It appears to require {macs} MACs + {memory} bytes + {params} params, "
            f"yielding cost = {cost} and score = {score:.3f}"
        )

'''
numpy version:  2.4.4
onnx version:  1.21.0
onnxruntime version:  1.24.4
onnx_tool version:  1.0.1
NEUROGOLF_FILE_VERSION:  28-apr.01

task102.onnx.onnx, simplify ok=True
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
Name    Type      Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
------  ------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
And_16  And                 81  2.69%             81  2.69%              0  0.00%       1x1x9x9    1x1x9x9
Not_0   Not                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_14  And                100  3.32%            100  3.32%              0  0.00%       1x1x10x10  1x1x10x10
And_23  And                 49  1.63%             49  1.63%              0  0.00%       1x1x7x7    1x1x7x7
And_9   And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_6   And                100  3.32%            100  3.32%              0  0.00%       1x1x10x10  1x1x10x10
Sub_0   Sub                144  4.79%            145  4.82%              1  100.00%     1          1x1x12x12
Max_0   Max                144  4.79%            144  4.79%              0  0.00%       1x1x12x12  1x1x12x12
Not_1   Not                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_2   And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_1   And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_7   And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_18  And                 64  2.13%             64  2.13%              0  0.00%       1x1x8x8    1x1x8x8
Add_0   Add                144  4.79%            144  4.79%              0  0.00%       1x1x12x12  1x1x12x12
And_15  And                 81  2.69%             81  2.69%              0  0.00%       1x1x9x9    1x1x9x9
And_10  And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_8   And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_0   And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
Not_2   Not                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_21  And                 49  1.63%             49  1.63%              0  0.00%       1x1x7x7    1x1x7x7
And_19  And                 64  2.13%             64  2.13%              0  0.00%       1x1x8x8    1x1x8x8
And_4   And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_5   And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_22  And                 49  1.63%             49  1.63%              0  0.00%       1x1x7x7    1x1x7x7
And_20  And                 64  2.13%             64  2.13%              0  0.00%       1x1x8x8    1x1x8x8
And_12  And                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
Not_3   Not                121  4.02%            121  4.02%              0  0.00%       1x1x11x11  1x1x11x11
And_13  And                100  3.32%            100  3.32%              0  0.00%       1x1x10x10  1x1x10x10
And_17  And                 81  2.69%             81  2.69%              0  0.00%       1x1x9x9    1x1x9x9
Total   _                3,008  100%           3,009  100%               1  100%        _          _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=     576    5.23% shape=[1, 1, 12, 12]
Cast_0_output                  bool       mem=     144    1.31% shape=[1, 1, 12, 12]
Cast_1_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Slice_1_output                 bool       mem=     121    1.10% shape=[1, 1, 11, 11]
Slice_2_output                 bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_0_C                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
Slice_3_output                 bool       mem=     121    1.10% shape=[1, 1, 11, 11]
Slice_4_output                 bool       mem=     121    1.10% shape=[1, 1, 11, 11]
Not_0_Y                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_1_C                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_2_C                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
Slice_5_output                 bool       mem=     100    0.91% shape=[1, 1, 10, 10]
Not_1_Y                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_4_C                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_5_C                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
Slice_6_output                 bool       mem=     100    0.91% shape=[1, 1, 10, 10]
And_6_C                        bool       mem=     100    0.91% shape=[1, 1, 10, 10]
Not_2_Y                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_7_C                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_8_C                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_9_C                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
Slice_7_output                 bool       mem=     100    0.91% shape=[1, 1, 10, 10]
Not_3_Y                        bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_10_C                       bool       mem=     121    1.10% shape=[1, 1, 11, 11]
And_12_C                       bool       mem=     121    1.10% shape=[1, 1, 11, 11]
Slice_8_output                 bool       mem=     100    0.91% shape=[1, 1, 10, 10]
And_13_C                       bool       mem=     100    0.91% shape=[1, 1, 10, 10]
And_14_C                       bool       mem=     100    0.91% shape=[1, 1, 10, 10]
Cast_2_output                  uint8      mem=     100    0.91% shape=[1, 1, 10, 10]
Pad_0_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Slice_9_output                 bool       mem=      81    0.74% shape=[1, 1, 9, 9]
Slice_10_output                bool       mem=      81    0.74% shape=[1, 1, 9, 9]
And_15_C                       bool       mem=      81    0.74% shape=[1, 1, 9, 9]
Slice_11_output                bool       mem=      81    0.74% shape=[1, 1, 9, 9]
Slice_12_output                bool       mem=      81    0.74% shape=[1, 1, 9, 9]
And_16_C                       bool       mem=      81    0.74% shape=[1, 1, 9, 9]
And_17_C                       bool       mem=      81    0.74% shape=[1, 1, 9, 9]
Cast_3_output                  uint8      mem=      81    0.74% shape=[1, 1, 9, 9]
Pad_1_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_2_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_3_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_4_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Slice_13_output                bool       mem=      64    0.58% shape=[1, 1, 8, 8]
Slice_14_output                bool       mem=      64    0.58% shape=[1, 1, 8, 8]
And_18_C                       bool       mem=      64    0.58% shape=[1, 1, 8, 8]
Slice_15_output                bool       mem=      64    0.58% shape=[1, 1, 8, 8]
Slice_16_output                bool       mem=      64    0.58% shape=[1, 1, 8, 8]
And_19_C                       bool       mem=      64    0.58% shape=[1, 1, 8, 8]
And_20_C                       bool       mem=      64    0.58% shape=[1, 1, 8, 8]
Cast_7_output                  uint8      mem=      64    0.58% shape=[1, 1, 8, 8]
Pad_5_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_6_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_7_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_8_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_9_output                   uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_10_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_11_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_12_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_13_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Slice_17_output                bool       mem=      49    0.45% shape=[1, 1, 7, 7]
Slice_18_output                bool       mem=      49    0.45% shape=[1, 1, 7, 7]
And_21_C                       bool       mem=      49    0.45% shape=[1, 1, 7, 7]
Slice_19_output                bool       mem=      49    0.45% shape=[1, 1, 7, 7]
Slice_20_output                bool       mem=      49    0.45% shape=[1, 1, 7, 7]
And_22_C                       bool       mem=      49    0.45% shape=[1, 1, 7, 7]
And_23_C                       bool       mem=      49    0.45% shape=[1, 1, 7, 7]
Cast_16_output                 uint8      mem=      49    0.45% shape=[1, 1, 7, 7]
Pad_14_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_15_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_16_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_17_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_18_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_19_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_20_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_21_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_22_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_23_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_24_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_25_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_26_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_27_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_28_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Pad_29_output                  uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Max_0_max                      uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Add_0_C                        uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Sub_0_C                        uint8      mem=     144    1.31% shape=[1, 1, 12, 12]
Concat_0_concat_result         uint8      mem=     864    7.85% shape=[1, 6, 12, 12]
output                         uint8      mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=   11010 100.00%

It appears to require 3008 MACs + 11010 bytes + 521 params, yielding cost = 14539 and score = 15.415
check: (3008, 11010, 521)

DONE !!!====================
len(examples) 267
wrong 0

'''
