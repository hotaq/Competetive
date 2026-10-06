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
taskname = "task114"


# don't add doctsting
'''
"""
TASK
arc-gen task generator: task_49d1d64f.json
Task:
Given a small input block (2×2, 2×3, 3×2, or 3×3) embedded in a padded 30×30 one-hot grid,
generate an edge-padded output (size up to 5×5) by extending rows/columns horizontally
and vertically (no diagonal corner filling), then embed back into 30×30.

ONNX SOLVER
Key Tricks & Hacks (for score optimization):
1. Unified gather-based construction:
   - Always crop 3×3 and decode via ArgMax → compact label space.
   - Use a single GatherElements pipeline with precomputed index tables.

2. flat11 trick (critical):
   - flat11 = [flat9, 255, 0]
     where:
       0..8 = real pixels
       9    = sentinel 255 (empty padding → all-zero after Equal)
       10   = true background (if needed)
   - Invalid regions directly index sentinel → removes need for masking.

3. Multi-case handling via index tables:
   - Four index tables: 2×2, 2×3, 3×2, 3×3.
   - Selection driven by simple size detection (row2/col2 presence).
   - Avoids duplicating the full computation graph.

4. Bool output (no Cast):
   - Use Equal(label, depth) directly.
   - Avoid Cast to uint8 → saves ~250–360 bytes.
   - Exploits Kaggle rule: output buffer not counted.

5. Compact working space:
   - Operate in 3×3 → 5×5 label space.
   - Only expand to 30×30 at the final Pad.

Result:
- Cost ≈ 1522
- Score ≈ 17.67
- Correct for all input sizes including 2×2.

Future Improvements:
- Replace chained Where-based table selection with a single table slice:
    case_id = has_row2 * 2 + has_col2
    idx = Gather(idx_all, case_id)
  This removes ~2 Where buffers (~200 bytes).
  Expected gain:
    score ≈ +0.05 to +0.10 (target ~17.75+).

- Further reduce size-detection overhead by eliminating Not/Cast chains
  via direct ReduceMax-based logic.

- Explore direct channel-space construction (bypassing label→onehot),
  though likely higher memory unless combined with lazy broadcast tricks.

Overall, this solution achieves a high score through compact label-space
operations, sentinel-based indexing, and minimal branching.
"""

'''

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

#####################################################################

def make_onnx():

    # don't modify
    reset_const_cache()
    SHAPE = (1, 10, 30, 30) #padH, padW
    input = argument(Tensor(np.float32, SHAPE))

    #========== start of your code here ============================
    # Crop largest possible source block: 3x3.
    crop = op_slice(input, [0, 0, 0, 0], [1, 10, 3, 3], [0, 1, 2, 3], [1, 1, 1, 1])

    # Decode one-hot crop to compact label map [1,1,3,3].
    lab3 = op.arg_max(crop, axis=1, keepdims=1)
    lab3 = op.cast(lab3, to=np.uint8)
    flat9 = op.reshape(lab3, op_constant([1, 1, 9], np.int64))

    # Append sentinel 255. 255 is never equal to any color depth 0..9,
    # so after Equal(depth) it becomes all-zero padding.
    flat11 = op.concat([flat9, op_constant(255, np.uint8, [1, 1, 1]), op_constant(0, np.uint8, [1, 1, 1])], axis=2)

    # Size detection.
    # Valid source colors are assumed non-black/nonzero. Padding is black/background 0.
    # has_row2 => H=3, has_col2 => W=3.
    row2 = op_slice(lab3, [0, 0, 2, 0], [1, 1, 3, 3], [0, 1, 2, 3], [1, 1, 1, 1])
    col2 = op_slice(lab3, [0, 0, 0, 2], [1, 1, 3, 3], [0, 1, 2, 3], [1, 1, 1, 1])

    row2_nonzero = op.not_(op.equal(row2, op_constant(0, np.uint8, [1, 1, 1, 3])))
    col2_nonzero = op.not_(op.equal(col2, op_constant(0, np.uint8, [1, 1, 3, 1])))

    row2_nonzero = op.cast(row2_nonzero, to=np.uint8)
    col2_nonzero = op.cast(col2_nonzero, to=np.uint8)

    has_row2 = op.reduce_max(row2_nonzero, axes=[2, 3], keepdims=0)
    has_col2 = op.reduce_max(col2_nonzero, axes=[2, 3], keepdims=0)
    has_row2 = op.reshape(has_row2, op_constant([1], np.int64))
    has_col2 = op.reshape(has_col2, op_constant([1], np.int64))

    # flat9 layout:
    #   0 1 2
    #   3 4 5
    #   6 7 8
    # index 9 = sentinel padding.
    # Output working canvas is always 5x5. Invalid cells point to 9.

    # 3x3 input -> 5x5 edge-padded without diagonal corners:
    # 9 0 1 2 9
    # 0 0 1 2 2
    # 3 3 4 5 5
    # 6 6 7 8 8
    # 9 6 7 8 9
    idx_3x3 = op_constant(np.array([
        10, 0, 1, 2, 10,
        0, 0, 1, 2, 2,
        3, 3, 4, 5, 5,
        6, 6, 7, 8, 8,
        10, 6, 7, 8, 10,
    ], dtype=np.int32), np.int32, [1, 1, 25])

    # 2x3 input -> active output is 4x5, embedded in 5x5:
    # 9 0 1 2 9
    # 0 0 1 2 2
    # 3 3 4 5 5
    # 9 3 4 5 9
    # 9 9 9 9 9
    idx_2x3 = op_constant(np.array([
        10, 0, 1, 2, 10,
        0, 0, 1, 2, 2,
        3, 3, 4, 5, 5,
        10, 3, 4, 5, 10,
        9, 9, 9, 9, 9,
    ], dtype=np.int32), np.int32, [1, 1, 25])

    # 3x2 input -> active output is 5x4, embedded in 5x5:
    # 9 0 1 9 9
    # 0 0 1 1 9
    # 3 3 4 4 9
    # 6 6 7 7 9
    # 9 6 7 9 9
    idx_3x2 = op_constant(np.array([
        10, 0, 1, 10, 9,
        0, 0, 1, 1, 9,
        3, 3, 4, 4, 9,
        6, 6, 7, 7, 9,
        10, 6, 7, 10, 9,
    ], dtype=np.int32), np.int32, [1, 1, 25])

    # 2x2 input -> active output is 4x4, embedded in 5x5:
    # 9 0 1 9 9
    # 0 0 1 1 9
    # 3 3 4 4 9
    # 9 3 4 9 9
    # 9 9 9 9 9
    idx_2x2 = op_constant(np.array([
        10, 0, 1, 10, 9,
        0, 0, 1, 1, 9,
        3, 3, 4, 4, 9,
        10, 3, 4, 10, 9,
        9, 9, 9, 9, 9,
    ], dtype=np.int32), np.int32, [1, 1, 25])

    # Select table:
    #   H=3,W=3 -> idx_3x3
    #   H=3,W=2 -> idx_3x2
    #   H=2,W=3 -> idx_2x3
    #   H=2,W=2 -> idx_2x2
    has_col2 = op.cast(has_col2, to=np.bool_)
    idx_h3 = op.where(has_col2, idx_3x3, idx_3x2)
    idx_h2 = op.where(has_col2, idx_2x3, idx_2x2)
    has_row2 = op.cast(has_row2, to=np.bool_)
    idx = op.where(has_row2, idx_h3, idx_h2)

    label25 = op.gather_elements(flat11, idx, axis=2)
    label5 = op.reshape(label25, op_constant([1, 1, 5, 5], np.int64))

    # Recover one-hot as bool; no uint8 cast, because output buffer is not counted and
    # bool one-hot compares correctly after the checker converts/compares arrays.
    depth = op_constant(np.arange(10), np.uint8, [1, 10, 1, 1])
    out5 = op.equal(label5, depth)

    # Pad compact 5x5 result to 30x30. Output memory is not counted by Kaggle profiler.
    output = op.pad(
        out5,
        pads=op_constant([0, 0, 0, 0, 0, 0, 25, 25], np.int64),
        constant_value=op_constant(False, dtype=np.bool_),
    )

    #========== end of your code here ============================
    model = spox.build(inputs={"input": input}, outputs={
        'output': output,
        #for debug

    })

    #debug_names(model)
    #debug_shapes(model)
    # model = patch_shapes(model,{
    #     'Pad_0_output': [1,10,30,30],
    # })

    return model

######################################################################################

#don't modify anything from this point onwards
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
        e = examples[i]

        inp = np.array(e["input"])
        inp = convert_color_to_onehot(inp)
        truth = np.array(e["output"])
        truth = convert_color_to_onehot(truth)

        (predict,) = session.run(["output"], {"input": inp})

        if 0:
            predict = session.run(debug_key, {'input': inp})
            for k, p in zip(debug_key, predict):
                print(k, p.shape)
                print(p)
            print()
            continue

        predict = predict.astype(np.float32)
        wng = 1 - np.array_equal(predict, truth)
        wrong += wng

        if (i < 2) or (wng):
            print(i, '------------------------------------')
            print("input")
            print_grid_color(convert_onehot_to_color(inp))
            print("predict")
            print_grid_color(convert_onehot_to_color(predict))
            print("truth")
            print_grid_color(convert_onehot_to_color(truth))
            print("predict[0,0]")
            print_grid_color(predict[0,0])

    print("DONE !!!====================")
    print("len(examples)", len(examples))
    print("wrong", wrong)


'''

numpy version:  2.4.4
onnx version:  1.21.0
onnxruntime version:  1.24.4
onnx_tool version:  1.0.1
NEUROGOLF_FILE_VERSION:  28-apr.01

task114.onnx.onnx, simplify ok=True
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
Name              Type              Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
----------------  --------------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
Where_2           Where                        0  0.00%            100  9.29%              0  0.00%       1          1x1x25
GatherElements_0  GatherElements               0  0.00%             25  2.32%              0  0.00%       1x1x11     1x1x25
Equal_2           Equal                      250  93.28%           260  24.14%            10  8.62%       1x1x5x5    1x10x5x5
Equal_0           Equal                        3  1.12%              6  0.56%              3  2.59%       1x1x1x3    1x1x1x3
Not_0             Not                          3  1.12%              3  0.28%              0  0.00%       1x1x1x3    1x1x1x3
ArgMax_0          ArgMax                       0  0.00%             72  6.69%              0  0.00%       1x10x3x3   1x1x3x3
Where_0           Where                        0  0.00%            300  27.86%            50  43.10%      1          1x1x25
ReduceMax_0       ReduceMax                    3  1.12%              1  0.09%              0  0.00%       1x1x1x3    1x1
Where_1           Where                        0  0.00%            300  27.86%            50  43.10%      1          1x1x25
Not_1             Not                          3  1.12%              3  0.28%              0  0.00%       1x1x3x1    1x1x3x1
Equal_1           Equal                        3  1.12%              6  0.56%              3  2.59%       1x1x3x1    1x1x3x1
ReduceMax_1       ReduceMax                    3  1.12%              1  0.09%              0  0.00%       1x1x3x1    1x1
Total             _                          268  100%           1,077  100%             116  100%        _          _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=     360   33.00% shape=[1, 10, 3, 3]
ArgMax_0_reduced               int64      mem=      72    6.60% shape=[1, 1, 3, 3]
Cast_0_output                  uint8      mem=       9    0.82% shape=[1, 1, 3, 3]
Reshape_0_reshaped             uint8      mem=       9    0.82% shape=[1, 1, 9]
Concat_0_concat_result         uint8      mem=      11    1.01% shape=[1, 1, 11]
Slice_1_output                 uint8      mem=       3    0.27% shape=[1, 1, 1, 3]
Equal_0_C                      bool       mem=       3    0.27% shape=[1, 1, 1, 3]
Not_0_Y                        bool       mem=       3    0.27% shape=[1, 1, 1, 3]
Cast_1_output                  uint8      mem=       3    0.27% shape=[1, 1, 1, 3]
ReduceMax_0_reduced            uint8      mem=       1    0.09% shape=[1, 1]
Reshape_1_reshaped             uint8      mem=       1    0.09% shape=[1]
Cast_2_output                  bool       mem=       1    0.09% shape=[1]
Slice_2_output                 uint8      mem=       3    0.27% shape=[1, 1, 3, 1]
Equal_1_C                      bool       mem=       3    0.27% shape=[1, 1, 3, 1]
Not_1_Y                        bool       mem=       3    0.27% shape=[1, 1, 3, 1]
Cast_3_output                  uint8      mem=       3    0.27% shape=[1, 1, 3, 1]
ReduceMax_1_reduced            uint8      mem=       1    0.09% shape=[1, 1]
Reshape_2_reshaped             uint8      mem=       1    0.09% shape=[1]
Cast_4_output                  bool       mem=       1    0.09% shape=[1]
Where_0_output                 int32      mem=     100    9.17% shape=[1, 1, 25]
Where_1_output                 int32      mem=     100    9.17% shape=[1, 1, 25]
Where_2_output                 int32      mem=     100    9.17% shape=[1, 1, 25]
GatherElements_0_output        uint8      mem=      25    2.29% shape=[1, 1, 25]
Reshape_3_reshaped             uint8      mem=      25    2.29% shape=[1, 1, 5, 5]
Equal_2_C                      bool       mem=     250   22.91% shape=[1, 10, 5, 5]
output                         bool       mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=    1091 100.00%

It appears to require 268 MACs + 1091 bytes + 163 params, yielding cost = 1522 and score = 17.672
check: (268, 1091, 163)

'''