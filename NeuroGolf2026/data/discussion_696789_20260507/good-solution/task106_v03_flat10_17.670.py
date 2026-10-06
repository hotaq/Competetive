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
taskname = "task106"




# don't add doctsting
"""
TASK
arc-gen task generator: task_46442a0e.py

Task:
Given a 2×2 or 3×3 input block (padded in a 30×30 one-hot grid), generate a rotated 2×2 tiling
(4×4 or 6×6) and embed it back into the padded 30×30 output.


ONNX SOLVER
Key Tricks & Hacks (for score optimization):
1. Unified gather pipeline:
   - Always crop a 3×3 region and use ArgMax → label map.
   - Handle 2×2 vs 3×3 via index selection instead of branching the graph.

2. flat10 trick (critical):
   - Build flat10 = [flat9, 255] where 255 is sentinel.
   - For 2×2 case, invalid cells directly index 9 → no need for masking.
   - Eliminates:
        - valid mask tensor
        - Where(valid, …)
        - extra Cast
   - Major memory reduction.

3. Bool output (no Cast):
   - Use Equal(label, depth) directly as output.
   - Avoid Cast to uint8 → saves ~360 bytes.
   - Exploits Kaggle rule: output buffer not counted.

4. Minimal dynamic logic:
   - Size detection via single pixel (2,2).
   - Replace Not + Equal with direct Equal-based selection.
   - Keep graph small and branch-free except one Where for index.

5. Work in compact space:
   - Operate in 3×3 → 6×6 → one-hot → final pad.
   - Avoid expanding to 30×30 until the last step.

Result:
- Cost ~1525
- Score ~17.67
- Near-optimal for this formulation.

Future Improvements:
- Explore eliminating final Equal (one-hot recovery) via direct channel scatter/gather,
  though this may increase buffer size.
- Investigate replacing index Where with arithmetic selection to remove another buffer.
- Try pure scatter-based construction of 7×7 label canvas to avoid 6×6 intermediate.
- Explore aggressive shape patching / lazy broadcast tricks for further memory hiding.

Overall, this solution balances correctness, clarity, and high NeuroGolf score using
compact label-space operations and sentinel-based indexing.
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

#####################################################################

def make_onnx():

    # don't modify
    reset_const_cache()
    SHAPE = (1, 10, 30, 30) #padH, padW
    input = argument(Tensor(np.float32, SHAPE))

    #========== start of your code here ============================
    # Crop the largest possible source block: 3x3.
    crop = op_slice(input, [0, 0, 0, 0], [1, 10, 3, 3], [0, 1, 2, 3], [1, 1, 1, 1])

    # Decode one-hot crop to label map: [1,1,3,3], uint8 colors 0..9.
    lab3 = op.arg_max(crop, axis=1, keepdims=1)
    lab3 = op.cast(lab3, to=np.uint8)

    # Detect input size.
    # Generator guarantee: real input block has no black/background pixels.
    # Therefore bottom-right of the 3x3 crop is nonzero iff the block is 3x3.
    br = op_slice(lab3, [0, 0, 2, 2], [1, 1, 3, 3], [0, 1, 2, 3], [1, 1, 1, 1])
    is2 = op.equal(br, op_constant(0, np.uint8, [1, 1, 1, 1]))
    is2 = op.reshape(is2, op_constant([1], np.int64))

    # Flatten 3x3 labels and append one sentinel value 255.
    # flat10 indices:
    #   0..8 = real/padded 3x3 cells
    #   9    = sentinel 255, later becomes all-zero after Equal(depth)
    flat9 = op.reshape(lab3, op_constant([1, 1, 9], np.int64))
    flat10 = op.concat([flat9, op_constant(255, np.uint8, [1, 1, 1])], axis=2)

    # Choose index table based on S.
    # 2x2 uses only source labels at flat9 positions:
    #   a=0, b=1, c=3, d=4
    # and writes index 9 into all cells outside the 4x4 active area.
    # This removes the previous valid-mask Where and label6 Where.
    idx2x2 = op_constant(np.array([
            0, 1, 3, 0, 9, 9,
            3, 4, 4, 1, 9, 9,
            1, 4, 4, 3, 9, 9,
            0, 3, 1, 0, 9, 9,
            9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9,
        ]), np.int32, [1, 1, 36])

    # 3x3 rotational 6x6 tile.
    idx3x3 = op_constant(np.array([
            0, 1, 2, 6, 3, 0,
            3, 4, 5, 7, 4, 1,
            6, 7, 8, 8, 5, 2,

            2, 5, 8, 8, 7, 6,
            1, 4, 7, 5, 4, 3,
            0, 3, 6, 2, 1, 0,
    ]), np.int32, [1, 1, 36])

    idx6 = op.where(is2, idx2x2, idx3x3)

    # Gather label canvas directly. For S=2, invalid cells already gather sentinel 255.
    tile36 = op.gather_elements(flat10, idx6, axis=2)
    label6 = op.reshape(tile36, op_constant([1, 1, 6, 6], np.int64))

    # Convert label map to one-hot 1x10x6x6.
    # Sentinel 255 compares false against all channel ids 0..9, producing all-zero padding.
    depth = op_constant(np.arange(10), np.uint8, [1, 10, 1, 1])
    out6  = op.equal(label6, depth)

    # Pad 6x6 one-hot canvas to final 30x30 output.
    output = op.pad(
        out6,
        pads=op_constant([0, 0, 0, 0, 0, 0, 24, 24], np.int64),
        constant_value=op_constant(0, dtype=np.bool),
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
        #print("i", i, "--------------------------------")
        e = examples[i]

        inp = np.array(e["input"])
        inp = convert_color_to_onehot(inp)
        truth = np.array(e["output"])
        truth = convert_color_to_onehot(truth)

        (predict,) = session.run(["output"], {"input": inp})
        #print("predict", predict.shape)
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

task106.onnx.onnx, simplify ok=True
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
Name              Type              Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
----------------  --------------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
ArgMax_0          ArgMax                       0  0.00%             72  7.89%              0  0.00%       1x10x3x3   1x1x3x3
Equal_1           Equal                      360  99.72%           370  40.57%            10  12.05%      1x1x6x6    1x10x6x6
GatherElements_0  GatherElements               0  0.00%             36  3.95%              0  0.00%       1x1x10     1x1x36
Equal_0           Equal                        1  0.28%              2  0.22%              1  1.20%       1x1x1x1    1x1x1x1
Where_0           Where                        0  0.00%            432  47.37%            72  86.75%      1          1x1x36
Total             _                          361  100%             912  100%              83  100%        _          _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=     360   34.65% shape=[1, 10, 3, 3]
ArgMax_0_reduced               int64      mem=      72    6.93% shape=[1, 1, 3, 3]
Cast_0_output                  uint8      mem=       9    0.87% shape=[1, 1, 3, 3]
Reshape_0_reshaped             uint8      mem=       9    0.87% shape=[1, 1, 9]
Concat_0_concat_result         uint8      mem=      10    0.96% shape=[1, 1, 10]
Slice_1_output                 uint8      mem=       1    0.10% shape=[1, 1, 1, 1]
Equal_0_C                      bool       mem=       1    0.10% shape=[1, 1, 1, 1]
Reshape_1_reshaped             bool       mem=       1    0.10% shape=[1]
Where_0_output                 int32      mem=     144   13.86% shape=[1, 1, 36]
GatherElements_0_output        uint8      mem=      36    3.46% shape=[1, 1, 36]
Reshape_2_reshaped             uint8      mem=      36    3.46% shape=[1, 1, 6, 6]
Equal_1_C                      bool       mem=     360   34.65% shape=[1, 10, 6, 6]
output                         bool       mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=    1039 100.00%

It appears to require 361 MACs + 1039 bytes + 125 params, yielding cost = 1525 and score = 17.670
check: (361, 1039, 125)

DONE !!!====================
len(examples) 266
wrong 0

'''
