import math
import numpy as np
import onnx
from onnxsim import simplify
import onnxruntime
from spox.opset.ai.onnx.v21 import cast

from my_local_helper import *

import spox
from spox import argument, Tensor
from spox.opset.ai.onnx import v17 as op

taskname = "task115"

'''
"""
TASK
arc-gen generator: task_4be741c5.py
Task:
Given a padded 30×30 one-hot grid containing colored regions, extract the ordered
sequence of unique colors along either the top edge (horizontal case) or left edge
(vertical case). Output the sequence as a compact 1×K or K×1 strip (K=3 or 4),
embedded back into a padded 30×30 one-hot grid.


ONNX SOLVER
Key Tricks & Hacks (for score optimization):
1. Edge-strip shortcut:
   - Only use top (1×10×1×30) and left (1×10×30×1) strips.
   - Avoid scanning full 30×30 grid.

2. Orientation via color count:
   - Use ReduceMax + ReduceSum on one-hot strips to count unique colors.
   - Compare top_count vs left_count to decide horizontal vs vertical.

3. Early label decoding:
   - Apply ArgMax separately on top and left strips.
   - Then select label strip via Where (size 30), avoiding large 10×30 buffers.

4. Fixed-shape “Compress” via TopK:
   - Build change mask: (label != prev) AND (label != 0).
   - Use TopK to select first 4 run-starts.
   - Tie-order ensures fixed shape output (length 4).

5. Sentinel padding:
   - Use 255 for unused slots (e.g., 3-color case).
   - After Equal(depth), sentinel becomes all-zero padding.

6. Scatter construction:
   - Build 4×4 canvas initialized with 255.
   - Scatter values into row or column positions depending on orientation.
   - Avoid building both horizontal and vertical canvases.

7. Bool one-hot output:
   - Use Equal(label, depth) directly (no Cast).
   - Output buffer is not counted by Kaggle profiler.

Result:
- Cost ≈ 5029
- Score ≈ 16.48
- Correct for all cases (266/266)

Limitations:
- Still materializes both top and left strips (float32 1200 bytes each).
- ArgMax produces int64 buffers (~240 bytes each).
- TopK introduces small overhead and relies on fixed max K=4.

Future Improvements:
- Avoid dual strip materialization by sampling fewer positions (generator-dependent).
- Replace ArgMax with weighted sum trick (may trade MACs vs memory).
- Reduce float32 strip memory by early casting or slicing fewer channels.
- Explore index-table or gather-based fixed logic to bypass TopK entirely.

Overall, this solution is efficient in logic (O(30)) and avoids full-grid operations,
but is limited by ONNX memory accounting of strip tensors.
"""

'''
############ helpers ################
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

def make_onnx():

    reset_const_cache()
    input = argument(Tensor(np.float32, (1,10,30,30)))

    # -------- strip extraction --------
    top  = op_slice(input, [0,0,0,0], [1,10,1,30], [0,1,2,3])
    left = op_slice(input, [0,0,0,0], [1,10,30,1], [0,1,2,3])

    # -------- count colors --------
    top_count  = op.reduce_max(top,  axes=[3])
    left_count = op.reduce_max(left, axes=[2])

    top_count  = op.reduce_sum(top_count,  axes=op_constant([0,1,2], np.int64), keepdims=False)
    left_count = op.reduce_sum(left_count, axes=op_constant([0,1,2], np.int64), keepdims=False)

    top_count  = op.cast(top_count,  to=np.uint8)
    left_count = op.cast(left_count, to=np.uint8)

    is_horizontal = op.greater(top_count, left_count)
    is_3 = op.equal(op.max([left_count, top_count]), op_constant([3], np.uint8))

    # -------- decode top/left separately, then select label strip --------
    # This avoids Where over 10x30 one-hot.
    top_label = op.arg_max(top, axis=1, keepdims=False)      # [1,1,30]
    left_label = op.arg_max(left, axis=1, keepdims=False)    # [1,30,1]

    top_label = op.cast(op.reshape(top_label, op_constant([30], np.int64)), to=np.uint8)
    left_label = op.cast(op.reshape(left_label, op_constant([30], np.int64)), to=np.uint8)

    is_horizontal_1d = op.reshape(is_horizontal, op_constant([1], np.int64))
    label = op.where(is_horizontal_1d, top_label, left_label)  # [30] uint8

    # -------- TopK as fixed-shape Compress replacement --------
    prev = op.concat([
        op_constant([29], np.uint8),
        op_slice(label, [0], [29], [0])
    ], axis=0)

    change = op.not_(op.equal(label, prev))
    nonzero = op.greater(label, op_constant(0, np.uint8))
    keep = op.and_(change, nonzero)

    # Keep your tie-order trick: shape fixed to 4.
    keep = op.cast(keep, to=np.float16)  #topk does not support uint8
    _, indices = op.top_k(keep, K=op_constant([4], np.int64))
    values = op.gather(label, indices, axis=0)

    # -------- scatter selected colors --------
    remove_last = op.where(
        is_3,
        op_constant([0,0,0,255], np.uint8),
        op_constant([0,0,0,0], np.uint8),
    )
    updates = op.max([values, remove_last])

    idx_h = op_constant(np.array([0,1,2,3]),  np.int32, [4])
    idx_v = op_constant(np.array([0,4,8,12]), np.int32, [4])
    scatter_idx = op.where(is_horizontal, idx_h, idx_v)

    out = op_constant(255, np.uint8, [16])
    out = op.scatter_elements(out, scatter_idx, updates, axis=0)
    out = op.reshape(out, op_constant([1,1,4,4], np.int64))

    # -------- one-hot recover --------
    depth = op_constant(np.arange(10), np.uint8, [1,10,1,1])
    out = op.equal(out, depth)
    output = op.pad(
        out,
        pads=op_constant([0,0,0,0,0,0,26,26], np.int64),
    )

    model = spox.build(inputs={"input": input}, outputs={"output": output})
    return model
#####################################################################################

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

        if 0:
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

    #---
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

task115.onnx.onnx, simplify ok=True
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
Name               Type               Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
-----------------  ---------------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
ReduceMax_0        ReduceMax                   300  33.08%            40  3.77%              0  0.00%       1x10x1x30  1x10x1x1
Max_0              Max                           1  0.11%              1  0.09%              0  0.00%       1          1
Equal_1            Equal                         1  0.11%              2  0.19%              1  2.08%       1          1
And_0              And                          30  3.31%             30  2.83%              0  0.00%       30         30
ScatterElements_0  ScatterElements               0  0.00%             32  3.02%             16  33.33%      16         16
ArgMax_1           ArgMax                        0  0.00%            240  22.62%             0  0.00%       1x10x30x1  1x30x1
Max_1              Max                           4  0.44%              4  0.38%              0  0.00%       4          4
Where_1            Where                         0  0.00%             30  2.83%              0  0.00%       1          30
Greater_0          Greater                       1  0.11%              1  0.09%              0  0.00%       1          1
Greater_1          Greater                      30  3.31%             31  2.92%              1  2.08%       30         30
Equal_0            Equal                        30  3.31%             30  2.83%              0  0.00%       30         30
Equal_2            Equal                       160  17.64%           170  16.02%            10  20.83%      1x1x4x4    1x10x4x4
Where_0            Where                         0  0.00%             48  4.52%              8  16.67%      1          4
ReduceSum_0        ReduceSum                    10  1.10%             28  2.64%              3  6.25%       1x10x1x1   1
TopK_0             TopK                          0  0.00%             48  4.52%              1  2.08%       30         4
ReduceSum_1        ReduceSum                    10  1.10%              4  0.38%              0  0.00%       1x10x1x1   1
ArgMax_0           ArgMax                        0  0.00%            240  22.62%             0  0.00%       1x10x1x30  1x1x30
ReduceMax_1        ReduceMax                   300  33.08%            40  3.77%              0  0.00%       1x10x30x1  1x10x1x1
Where_2            Where                         0  0.00%             12  1.13%              8  16.67%      1          4
Not_0              Not                          30  3.31%             30  2.83%              0  0.00%       30         30
Total              _                           907  100%           1,061  100%              48  100%        _          _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=    1200   29.69% shape=[1, 10, 1, 30]
ReduceMax_0_reduced            float32    mem=      40    0.99% shape=[1, 10, 1, 1]
ReduceSum_0_reduced            float32    mem=       4    0.10% shape=[1]
Cast_0_output                  uint8      mem=       1    0.02% shape=[1]
Slice_1_output                 float32    mem=    1200   29.69% shape=[1, 10, 30, 1]
ReduceMax_1_reduced            float32    mem=      40    0.99% shape=[1, 10, 1, 1]
ReduceSum_1_reduced            float32    mem=       4    0.10% shape=[1]
Cast_1_output                  uint8      mem=       1    0.02% shape=[1]
Greater_0_C                    bool       mem=       1    0.02% shape=[1]
Where_0_output                 int32      mem=      16    0.40% shape=[4]
ArgMax_0_reduced               int64      mem=     240    5.94% shape=[1, 1, 30]
Reshape_1_reshaped             int64      mem=     240    5.94% shape=[30]
Cast_2_output                  uint8      mem=      30    0.74% shape=[30]
ArgMax_1_reduced               int64      mem=     240    5.94% shape=[1, 30, 1]
Reshape_2_reshaped             int64      mem=     240    5.94% shape=[30]
Cast_3_output                  uint8      mem=      30    0.74% shape=[30]
Where_1_output                 uint8      mem=      30    0.74% shape=[30]
Slice_2_output                 uint8      mem=      29    0.72% shape=[29]
Concat_0_concat_result         uint8      mem=      30    0.74% shape=[30]
Equal_0_C                      bool       mem=      30    0.74% shape=[30]
Not_0_Y                        bool       mem=      30    0.74% shape=[30]
Greater_1_C                    bool       mem=      30    0.74% shape=[30]
And_0_C                        bool       mem=      30    0.74% shape=[30]
Cast_4_output                  float16    mem=      60    1.48% shape=[30]
TopK_0_Values                  float16    mem=       8    0.20% shape=[4]
TopK_0_Indices                 int64      mem=      32    0.79% shape=[4]
Gather_0_output                uint8      mem=       4    0.10% shape=[4]
Max_0_max                      uint8      mem=       1    0.02% shape=[1]
Equal_1_C                      bool       mem=       1    0.02% shape=[1]
Where_2_output                 uint8      mem=       4    0.10% shape=[4]
Max_1_max                      uint8      mem=       4    0.10% shape=[4]
ScatterElements_0_output       uint8      mem=      16    0.40% shape=[16]
Reshape_3_reshaped             uint8      mem=      16    0.40% shape=[1, 1, 4, 4]
Equal_2_C                      bool       mem=     160    3.96% shape=[1, 10, 4, 4]
output                         bool       mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=    4042 100.00%

It appears to require 907 MACs + 4042 bytes + 80 params, yielding cost = 5029 and score = 16.477
check: (907, 4042, 80)

DONE !!!====================
len(examples) 266
wrong 0

'''