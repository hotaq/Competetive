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
taskname = "task117"

# don't add doctsting
"""


TASK
arc-gen task generator: task_4c5c2cf0.py

Detect the central X-shaped marker and mirror the surrounding 
arm pixels across it to complete all four symmetric quadrants.


ONNX SOLVER

Task117 ONNX solution (NeuroGolf optimized, ~14.95 score)
Core idea:
----------
This solution exploits symmetry and avoids explicit case detection.
Instead of identifying which quadrant contains the arm, it performs
full symmetric expansion around the detected center.

Algorithm overview:
------------------
1) Crop input to 15x15 working grid.

2) Detect valid region using background channel only:
   - Use top row and left column of bg channel to infer true grid.
   - Avoid expensive full-channel reductions.

3) Decode label map:
   - ArgMax once → [1,1,15,15] uint8.
   - Keep padded area as sentinel (255).

4) Detect center (X-shape):
   - Restrict search to 5x5 window (positions 5..9).
   - Use local equality checks instead of global scans.
   - Compute (cy, cx) via weighted sum (no loops).

5) Symmetric expansion (key trick):
   - Do NOT detect quadrant.
   - Generate 4 symmetric versions:
        original
        vertical reflection
        horizontal reflection
        both reflections
   - Use 1D Gather for reflection (no GatherND).
   - Merge using op.max([...]) to fill all quadrants.

6) Boundary handling:
   - Use small validity masks (Less + Where).
   - Cheaper than padding large tensors.

7) Final output:
   - Restore sentinel outside valid region.
   - Convert to one-hot using Equal(depth).
   - Pad back to 30x30.

Key optimizations:
-----------------
- Geometry-first: operate on compact label map instead of 10 channels.
- Single ArgMax only.
- Avoid GatherND (major memory saver).
- Use uint8 for all heavy tensors.
- Restrict center search to 5x5 instead of 13x13.
- Symmetry expansion replaces branching logic.
- Boundary checks are cheaper than padding.
- Variadic op.max([...]) reduces intermediate buffers.

Performance:
------------
MACs:   ~3700
Memory: ~18800 bytes
Params: ~600
Score:  ~14.95

Notes:
------
- Final Equal (one-hot expansion) is the dominant cost and hard to avoid.
- Further gains likely require avoiding full 10-channel materialization.
- This solution is stable and avoids risky hacks.

Common pitfalls avoided:
-----------------------
- op.max must use list: op.max([a,b]), not op.max(a,b)
- Gather indices must be int64
- Avoid rank explosion from improper broadcasting
- Avoid large Where cascades and GatherND

Overall:
--------
A balanced, high-scoring solution using symmetry + minimal tensors.
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
    # v10: v09 baseline plus fused four-way Max.
    # Only the final safe indices are cast to int64 for Gather, reducing the
    # small int64 index-vector buffers and arithmetic cost.

    # -------- crop, valid mask, and decode --------
    crop15 = op_slice(input, [0,0,0,0], [1,10,15,15], [0,1,2,3])

    # Valid square detection from bg channel strips.
    # True ARC grid is SxS, S in {12,13,14,15}; outside SxS is all-zero onehot.
    bg15 = op_slice(input, [0,0,0,0], [1,1,15,15], [0,1,2,3])
    top_valid = op.cast(
        op_slice(bg15, [0,0,0,0], [1,1,1,15], [0,1,2,3]),
        to=np.bool,
    )                                                       # [1,1,1,15]
    left_valid = op.cast(
        op_slice(bg15, [0,0,0,0], [1,1,15,1], [0,1,2,3]),
        to=np.bool,
    )                                                       # [1,1,15,1]
    valid15 = op.and_(left_valid, top_valid)                # [1,1,15,15]

    lab_raw15 = op.cast(op.arg_max(crop15, axis=1), to=np.uint8)

    zero_i64 = op_constant(0, np.int64)
    zero = op_constant(0, np.uint8)
    two  = op_constant(2, np.uint8)
    c15  = op_constant(15, np.uint8)
    sent = op_constant(255, np.uint8)
    depth = op_constant(np.arange(10), np.uint8, [1,10,1,1])

    lab15 = op.where(valid15, lab_raw15, op_constant(255, np.uint8, [1,1,15,15]))

    # -------- find 3x3 X center, vectorized over the constrained 5x5 window --------
    # Candidate true centers: y/x = 5..9.  All tensors here are only [1,1,5,5].
    ctr = op_slice(lab15, [0,0,5,5],  [1,1,10,10], [0,1,2,3])
    tl  = op_slice(lab15, [0,0,4,4],  [1,1,9,9],   [0,1,2,3])
    tr  = op_slice(lab15, [0,0,4,6],  [1,1,9,11],  [0,1,2,3])
    bl  = op_slice(lab15, [0,0,6,4],  [1,1,11,9],  [0,1,2,3])
    br  = op_slice(lab15, [0,0,6,6],  [1,1,11,11], [0,1,2,3])
    tm  = op_slice(lab15, [0,0,4,5],  [1,1,9,10],  [0,1,2,3])
    ml  = op_slice(lab15, [0,0,5,4],  [1,1,10,9],  [0,1,2,3])
    mr  = op_slice(lab15, [0,0,5,6],  [1,1,10,11], [0,1,2,3])
    bm  = op_slice(lab15, [0,0,6,5],  [1,1,11,10], [0,1,2,3])

    nonzero = op.and_(op.not_(op.equal(ctr, zero)), op.not_(op.equal(ctr, sent)))
    m = op.and_(nonzero, op.equal(ctr, tl))
    m = op.and_(m,       op.equal(ctr, tr))
    m = op.and_(m,       op.equal(ctr, bl))
    m = op.and_(m,       op.equal(ctr, br))
    # Cheap local negative checks; no full exact-5 color scan.
    m = op.and_(m,       op.not_(op.equal(ctr, tm)))
    m = op.and_(m,       op.not_(op.equal(ctr, ml)))
    m = op.and_(m,       op.not_(op.equal(ctr, mr)))
    m = op.and_(m,       op.not_(op.equal(ctr, bm)))
    mi = op.cast(m, to=np.int64)                            # [1,1,5,5]

    yy5 = op_constant(np.arange(5,10, dtype=np.int64).reshape(1,1,5,1), np.int64)
    xx5 = op_constant(np.arange(5,10, dtype=np.int64).reshape(1,1,1,5), np.int64)
    cy = op.reduce_sum(op.mul(mi, yy5), axes=op_constant([0,1,2,3], np.int64), keepdims=False)  # scalar
    cx = op.reduce_sum(op.mul(mi, xx5), axes=op_constant([0,1,2,3], np.int64), keepdims=False)  # scalar

    # -------- mirror render with dynamic 1D Gather, not GatherND --------
    # Use 1D row/col index vectors.  This avoids the expensive [1,1,15,15,4]
    # GatherND index tensors and avoids v07's 5-case Where cascades.
    src15 = op.where(op.equal(lab15, sent), zero, lab15)      # uint8, padded area ignored

    # Compute mirrored row/column indices in uint8.  Negative values wrap to
    # large uint8 values, so the only validity check needed is idx < 15.
    # Gather still requires int64 indices, so cast only ysafe/xsafe at the end.
    cy_u8 = op.cast(cy, to=np.uint8)
    cx_u8 = op.cast(cx, to=np.uint8)
    ar15_u8 = op_constant(np.arange(15, dtype=np.uint8), np.uint8)  # [15]
    yidx_u8 = op.sub(op.mul(two, cy_u8), ar15_u8)                  # [15], uint8
    xidx_u8 = op.sub(op.mul(two, cx_u8), ar15_u8)                  # [15], uint8


    #----------
    yvalid1 = op.less(yidx_u8, c15)
    xvalid1 = op.less(xidx_u8, c15)
    ysafe_u8 = op.where(yvalid1, yidx_u8, op_constant(0, np.uint8, [15]))
    xsafe_u8 = op.where(xvalid1, xidx_u8, op_constant(0, np.uint8, [15]))
    ysafe = op.cast(ysafe_u8, to=np.int64)
    xsafe = op.cast(xsafe_u8, to=np.int64)

    ymask = op.reshape(yvalid1, op_constant([1,1,15,1], np.int64))
    xmask = op.reshape(xvalid1, op_constant([1,1,1,15], np.int64))

    v00 = src15
    v10 = op.gather(src15, ysafe, axis=2)                    # reflected rows
    v10 = op.where(ymask, v10, op_constant(0, np.uint8, [1,1,15,15]))

    v01 = op.gather(src15, xsafe, axis=3)                    # reflected cols
    v01 = op.where(xmask, v01, op_constant(0, np.uint8, [1,1,15,15]))

    v11 = op.gather(v10, xsafe, axis=3)                      # both reflections
    v11 = op.where(xmask, v11, op_constant(0, np.uint8, [1,1,15,15]))
    #----------


    # v10: fuse four mirror candidates into one variadic Max to remove two intermediate 15x15 buffers.
    out15_raw = op.max([v00, v10, v01, v11])
    out15 = op.where(valid15, out15_raw, op_constant(255, np.uint8, [1,1,15,15]))

    # -------- one-hot recover --------
    out = op.equal(out15, depth)

    output = op.pad(
        out,
        pads=op_constant([0,0,0,0,0,0,15,15], np.int64),
    )
    #ouput can be bool, uint8, etc

    #========== end of your code here ============================

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

    #- alst check ------------
    if 0:
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
/home/hp/app/anaconda3.11-cv/bin/python /media/hp/8TB-HDD/work/2026/kaggle/neurogolf/code/dummy01/hand-solution/task117/task117_v10a_from_v09_14.948.py 
2026-05-03 17:51:20.419430338 [W:onnxruntime:Default, device_discovery.cc:325 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:92 ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
numpy version:  2.4.4
onnx version:  1.21.0
onnxruntime version:  1.24.4
onnx_tool version:  1.0.1
NEUROGOLF_FILE_VERSION:  28-apr.01

task117.onnx.onnx, simplify ok=True
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
Name         Type         Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape     OutShape
-----------  ---------  --------------  ----------  --------  ----------  --------  ----------  ----------  ----------
Equal_4      Equal                  25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Equal_7      Equal                  25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Not_4        Not                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Mul_0        Mul                    25  0.67%            240  3.08%              5  0.98%       1x1x5x5     1x1x5x5
Mul_3        Mul                     1  0.03%              2  0.03%              1  0.20%       1           1
Where_4      Where                   0  0.00%             30  0.38%             15  2.95%       15          15
And_8        And                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Equal_5      Equal                  25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Where_7      Where                   0  0.00%            450  5.77%            225  44.29%      1x1x15x15   1x1x15x15
And_3        And                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Where_0      Where                   0  0.00%            225  2.89%              0  0.00%       1x1x15x15   1x1x15x15
Less_0       Less                   15  0.40%             16  0.21%              1  0.20%       15          15
Equal_1      Equal                  25  0.67%             26  0.33%              1  0.20%       1x1x5x5     1x1x5x5
Where_6      Where                   0  0.00%            450  5.77%            225  44.29%      1x1x1x15    1x1x15x15
Where_3      Where                   0  0.00%            225  2.89%              0  0.00%       1x1x15x1    1x1x15x15
ReduceSum_1  ReduceSum              25  0.67%             32  0.41%              4  0.79%       1x1x5x5     0
Equal_8      Equal                  25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
And_4        And                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
And_7        And                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Sub_1        Sub                    15  0.40%             30  0.38%             15  2.95%       1           15
Equal_9      Equal                  25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Equal_11     Equal               2,250  60.61%         2,260  28.98%            10  1.97%       1x1x15x15   1x10x15x15
Less_1       Less                   15  0.40%             15  0.19%              0  0.00%       15          15
Not_0        Not                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Not_5        Not                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Mul_2        Mul                    25  0.67%            240  3.08%              5  0.98%       1x1x5x5     1x1x5x5
Equal_10     Equal                  25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
ReduceSum_0  ReduceSum              25  0.67%              0  0.00%              0  0.00%       1x1x5x5     0
Max_0        Max                   225  6.06%            225  2.89%              0  0.00%       1x1x15x15   1x1x15x15
And_2        And                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Equal_2      Equal                  25  0.67%             26  0.33%              1  0.20%       1x1x5x5     1x1x5x5
Mul_1        Mul                     1  0.03%              1  0.01%              0  0.00%       1           1
Equal_6      Equal                  25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
And_0        And                   225  6.06%            225  2.89%              0  0.00%       1x1x15x1    1x1x15x15
Equal_3      Equal                  25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Not_1        Not                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
And_9        And                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Equal_0      Equal                 225  6.06%            225  2.89%              0  0.00%       1x1x15x15   1x1x15x15
Sub_0        Sub                    15  0.40%             15  0.19%              0  0.00%       1           15
Not_2        Not                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
And_1        And                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
ArgMax_0     ArgMax                  0  0.00%          1,800  23.08%             0  0.00%       1x10x15x15  1x1x15x15
Where_1      Where                   0  0.00%            225  2.89%              0  0.00%       1x1x15x15   1x1x15x15
And_5        And                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
And_6        And                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Where_2      Where                   0  0.00%             15  0.19%              0  0.00%       15          15
Where_5      Where                   0  0.00%            225  2.89%              0  0.00%       1x1x1x15    1x1x15x15
Not_3        Not                    25  0.67%             25  0.32%              0  0.00%       1x1x5x5     1x1x5x5
Total        _                   3,712  100%           7,798  100%             508  100%        _           _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=     900    4.77% shape=[1, 1, 15, 15]
Slice_1_output                 float32    mem=      60    0.32% shape=[1, 1, 15, 1]
Cast_0_output                  bool       mem=      15    0.08% shape=[1, 1, 15, 1]
Slice_2_output                 float32    mem=      60    0.32% shape=[1, 1, 1, 15]
Cast_1_output                  bool       mem=      15    0.08% shape=[1, 1, 1, 15]
And_0_C                        bool       mem=     225    1.19% shape=[1, 1, 15, 15]
Slice_3_output                 float32    mem=    9000   47.73% shape=[1, 10, 15, 15]
ArgMax_0_reduced               int64      mem=    1800    9.55% shape=[1, 1, 15, 15]
Cast_2_output                  uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Where_0_output                 uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Equal_0_C                      bool       mem=     225    1.19% shape=[1, 1, 15, 15]
Where_1_output                 uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Slice_4_output                 uint8      mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_1_C                      bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Not_0_Y                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_2_C                      bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Not_1_Y                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
And_1_C                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Slice_5_output                 uint8      mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_3_C                      bool       mem=      25    0.13% shape=[1, 1, 5, 5]
And_2_C                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Slice_6_output                 uint8      mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_4_C                      bool       mem=      25    0.13% shape=[1, 1, 5, 5]
And_3_C                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Slice_7_output                 uint8      mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_5_C                      bool       mem=      25    0.13% shape=[1, 1, 5, 5]
And_4_C                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Slice_8_output                 uint8      mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_6_C                      bool       mem=      25    0.13% shape=[1, 1, 5, 5]
And_5_C                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Slice_9_output                 uint8      mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_7_C                      bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Not_2_Y                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
And_6_C                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Slice_10_output                uint8      mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_8_C                      bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Not_3_Y                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
And_7_C                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Slice_11_output                uint8      mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_9_C                      bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Not_4_Y                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
And_8_C                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Slice_12_output                uint8      mem=      25    0.13% shape=[1, 1, 5, 5]
Equal_10_C                     bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Not_5_Y                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
And_9_C                        bool       mem=      25    0.13% shape=[1, 1, 5, 5]
Cast_3_output                  int64      mem=     200    1.06% shape=[1, 1, 5, 5]
Mul_0_C                        int64      mem=     200    1.06% shape=[1, 1, 5, 5]
ReduceSum_0_reduced            int64      mem=       8    0.04% shape=[]
Cast_4_output                  uint8      mem=       1    0.01% shape=[]
Mul_1_C                        uint8      mem=       1    0.01% shape=[1]
Sub_0_C                        uint8      mem=      15    0.08% shape=[15]
Less_0_C                       bool       mem=      15    0.08% shape=[15]
Reshape_0_reshaped             bool       mem=      15    0.08% shape=[1, 1, 15, 1]
Where_2_output                 uint8      mem=      15    0.08% shape=[15]
Cast_5_output                  int64      mem=     120    0.64% shape=[15]
Gather_0_output                uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Where_3_output                 uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Mul_2_C                        int64      mem=     200    1.06% shape=[1, 1, 5, 5]
ReduceSum_1_reduced            int64      mem=       8    0.04% shape=[]
Cast_6_output                  uint8      mem=       1    0.01% shape=[]
Mul_3_C                        uint8      mem=       1    0.01% shape=[1]
Sub_1_C                        uint8      mem=      15    0.08% shape=[15]
Less_1_C                       bool       mem=      15    0.08% shape=[15]
Reshape_1_reshaped             bool       mem=      15    0.08% shape=[1, 1, 1, 15]
Where_4_output                 uint8      mem=      15    0.08% shape=[15]
Cast_7_output                  int64      mem=     120    0.64% shape=[15]
Gather_1_output                uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Where_5_output                 uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Gather_2_output                uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Where_6_output                 uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Max_0_max                      uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Where_7_output                 uint8      mem=     225    1.19% shape=[1, 1, 15, 15]
Equal_11_C                     bool       mem=    2250   11.93% shape=[1, 10, 15, 15]
output                         bool       mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=   18855 100.00%

It appears to require 3712 MACs + 18855 bytes + 608 params, yielding cost = 23175 and score = 14.949
check: (3712, 18855, 608)

DONE !!!====================
len(examples) 265
wrong 0


Process finished with exit code 0
 


'''