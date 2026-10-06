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
taskname = "task107"

# don't add doctsting
"""
TASK
arc-gen task generator task_469497ad.py
NeuroGolf Task107 – Final Solution (v13)
Input is upscaled by number of colors. Please find a solid colored square and emits diagonal red rays from its corners.


ONNX SOLVER
This solution implements a minimal-cost pipeline by decoupling scaling, drawing,
and layout transformation:

1. Decode 5×5 grid:
   Convert one-hot input to label grid via ArgMax.

2. Determine scale s (2..6):
   Count unique colors in the 5×5 grid using ReduceMax + ReduceSum.

3. Detect box position:
   Use two probe pixels (p01, p10) to derive (sy, sx) and encode into a
   compact index si ∈ {0,1,2}.

4. Expand to maximum resolution:
   Reshape + Expand transforms 5×5 → 5×6×5×6 (implicit 30×30 layout)
   without any index table or Gather.

5. Draw red diagonals (max scale s=6):
   Use a compact precomputed table of 24 indices per position and
   ScatterElements to overwrite red pixels efficiently.

6. Shrink back to true scale:
   Apply two 1D Gather operations (row then column) using a small
   30-entry index table per scale. This performs block compaction
   from 6×6 to s×s while preserving diagonal alignment.

7. Final one-hot:
   Convert label grid to output channels via Equal.

Key optimizations:
- Avoids large 900-element gather/scatter index tables.
- Uses Expand instead of Gather for upscaling.
- Uses compact 24-index scatter for red drawing.
- Uses 1D gather (size 30) instead of 2D mapping (size 900).
- Parameters reduced to ~315.

Performance:
- ~9268 MACs, ~8993 bytes memory, ~315 params
- Score ≈ 15.17

Limitation:
- Final Equal (one-hot) dominates compute (~9000 MACs) and is the main
  remaining bottleneck.

Overall, this design reaches near-optimal efficiency for this task by
minimizing index-based operations and exploiting structured layout.
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

# added helpers only

def i8_const(value, shape=None):
    return op_constant(value, np.int8, shape)

def u8_const(value, shape=None):
    return op_constant(value, np.uint8, shape)

def i32_const(value, shape=None):
    return op_constant(value, np.int32, shape)

def i64_const(value, shape=None):
    return op_constant(value, np.int64, shape)

def bool_const(value, shape=None):
    return op_constant(value, np.bool_, shape)


def make_onnx():

    # don't modify
    reset_const_cache()
    SHAPE = (1, 10, 30, 30) #padH, padW
    input = argument(Tensor(np.float32, SHAPE))

    #========== start of your code here ============================

    crop5x5 = op_slice(input, [0, 0, 0, 0], [1, 10, 5, 5], [0, 1, 2, 3], [1, 1, 1, 1])
    x5 = op.arg_max(crop5x5, axis=1, keepdims=1)
    x5 = op.cast(x5, to=np.uint8)

    # count colors in whole 5x5.  s = unique non-bg/non-red-ish count + 1 via existing onehot count rule.
    count = op.reduce_max(crop5x5, axes=[0, 2, 3], keepdims=0)
    count = op.reduce_sum(count)
    s = op.sub(op.cast(count, to=np.uint8), u8_const([1]))  # values 2..6 in observed data

    # box position probes.
    '''
    bg
    
    11
    10  sy,sx=(1,1) p01=1,p10=1, si=2
    
    10
    10  sy,sx=(0,1) p01=0,p10=1, si=1
    
    11
    00  sy,sx=(1,0) p01=1,p10=0, si=0  
    
    '''

    p01 = op_slice(crop5x5, [0, 0, 1], [1, 1, 2], [1, 2, 3], [ 1, 1, 1])
    p10 = op_slice(crop5x5, [0, 1, 0], [1, 2, 1], [1, 2, 3], [ 1, 1, 1])
    sy = op.reshape( op.cast( p01, to=np.uint8), i64_const([1]))
    sx = op.reshape( op.cast( p10, to=np.uint8), i64_const([1]))
    si = op.add(sx, op.mul(sx,sy))
    si = op.cast( si, to=np.int64)

    # make the largest s: 5x5 -> 5x6x5x6 -> 30x30 block layout.
    block = op.reshape(x5, i64_const([5, 1, 5, 1]))
    block = op.expand(block, i64_const([5, 6, 5, 6]))

    # --- make red  digonal ---
    def make_red_digonal():
        s=6
        index = []
        for sy,sx in[
            (1,0),(0,1),(1,1),
        ]:
            idx =[]
            for by,bx, direction in[
                (-1,-1, 1),(-1,2, 0),(2,-1, 0),(2,2, 1)
            ]:
                by = (by+sy)*s
                bx = (bx+sx)*s
                if (by>=0) & (bx>=0):
                    for y in range(s):
                        if (direction==1):
                            idx.append((by+y)*30+bx+y)
                        if (direction==0):
                            idx.append((by+y)*30+bx+(6-y-1))
            idx = np.asarray(idx, dtype=np.int32)
            #print(idx.shape)  # (24,),(12,),(12,)
            if len(idx)!=24:
                idx = np.concat(
                    [idx, np.full(shape=24-len(idx), fill_value=idx[-1], dtype=np.int32)], axis=0
                )
            index.append(idx)
        index = np.stack(index, axis=0).reshape(-1)
        #print(index.shape)
        return index

    red_index = i32_const(make_red_digonal())
    starts = op.mul(si, i64_const([24]))
    ends = op.add(starts, i64_const([24]))
    red_idx = op.reshape(
        op.slice(red_index, starts, ends),
        shape=op_constant([24], np.int64)
    )
    red = op_constant(2,dtype=np.uint8, shape=[24])

    block = op.reshape(block, i64_const([900]))
    block = op.scatter_elements(
        block,
        red_idx, #can be int32 or int64
        red,
        axis=0,
    )

    #resize back to sxs ------------------------------------------
    def make_shrink_block():
        # def map_array(idx, n, m):
        #     return idx + m * (idx // n)

        mapping=[
            [
                0,1,           6,7,                         16,17,             22,23,              28,29,
            ],[
                0,1,2,         6,7,8,                    15,16,17,           21,22,23,           27,28,29,
            ],[
                0,1,2,3,       6,7,8,9,               14,15,16,17,        20,21,22,23,        26,27,28,29,
            ],[
                0,1,2,3,4,     6,7,8,9,10,         13,14,15,16,17,     19,20,21,22,23,     25,26,27,28,29,
            ],[
                0,1,2,3,4,5,   6,7,8,9,10,11,   12,13,14,15,16,17,  18,19,20,21,22,23,  24,25,26,27,28,29,
            ],
        ]

        index = []
        for s in[
            2,3,4,5,6
        ]:
            idx = np.array(mapping[s-2])
            #print(s, idx.shape)

            idx = np.concat([
                idx, np.full(shape=30-len(idx), fill_value=30, dtype=np.int32)
            ])
            index.append(idx)
        index = np.stack(index, axis=0).reshape(-1)
        #print('index', index.shape)
        #print(index)
        return index

    shrink_index = i32_const(make_shrink_block())
    starts = op.mul(
        op.sub(op.cast(s,to=np.int32),i32_const([2])),
        i32_const([30]))
    ends  = op.add(starts, i32_const([30]))
    shrink_idx = op.reshape(
        op.slice( shrink_index, starts, ends),
        shape=op_constant([30], np.int64)
    )
    shrink_idx = op.reshape(shrink_idx, shape=i64_const([30]))
    block30x30 = op.reshape(block, i64_const([30,30]))
    block31x31 = op.pad(block30x30,pads=i64_const([0,0,1,1]),constant_value=u8_const([255]))
    shrink = op.gather(
        block31x31,
        shrink_idx, #can be int32 or int64
        axis=0  # row axis
    )
    shrink = op.gather(
        shrink,
        shrink_idx,
        axis=1  # row axis
    )


    # pad to onehot and output!
    label30x30 = op.reshape(shrink, i64_const([1, 1, 30, 30]))
    output = op.equal(label30x30, u8_const(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))

    ##############################################################

    model = spox.build(inputs={"input": input}, outputs={
        'output': output,
    })

    # Dynamic Slice lengths are known by construction, but onnx_tool may infer [0].
    debug_shapes(model)
    # model = patch_shapes(model, {
    #      #'Slice_3_output': [24],
    #      #'Gather_1_output': [30,30],
    # })
    return model

######################################################################################

#don't modify anything after this
if __name__ == "__main__":
    onnx_file = f"{taskname}.onnx"
    memory = print_kaggle_memory_proile(f"{taskname}-fix1.onnx")

    model = onnx.load(f"{taskname}-fix1.onnx")
   # model.graph.ClearField("value_info")
    model = onnx.shape_inference.infer_shapes(model)

    for v in list(model.graph.value_info) + list(model.graph.output):
        t = v.type.tensor_type
        if not t.HasField("shape"):
            continue
        dims = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append("?")
        #if any(not isinstance(x, int) for x in dims):
        if 1:
            print(v.name, "->", dims)

    exit(0)

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

    debug_key=[]
    for out in model.graph.output:
        if out.name!='output':
            debug_key.append(out.name)

    wrong = 0
    for i in range(len(examples)):
        print("i", i, "--------------------------------")
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
        print("predict", predict.shape)

        predict = predict.astype(np.float32)
        wng = 1 - np.array_equal(predict, truth)
        wrong += wng

        if (i < 5): # or (wng):
            print("input")
            print_grid_color(convert_onehot_to_color(input))
            print("predict")
            print_grid_color(convert_onehot_to_color(predict))
            print("truth")
            print_grid_color(convert_onehot_to_color(truth))

    print("DONE !!!====================")
    print("len(examples)", len(examples))
    print("wrong", wrong)


'''

/home/hp/app/anaconda3.11-cv/bin/python /media/hp/8TB-HDD/work/2026/kaggle/neurogolf/code/dummy01/hand-solution/task107/task107_v13.py 
2026-05-02 22:51:10.849399359 [W:onnxruntime:Default, device_discovery.cc:325 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:92 ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
numpy version:  2.4.4
onnx version:  1.21.0
onnxruntime version:  1.24.4
onnx_tool version:  1.0.1
NEUROGOLF_FILE_VERSION:  28-apr.01

2 (10,)
3 (15,)
4 (20,)
5 (25,)
6 (30,)
index (150,)
[ 0  1  6  7 16 17 22 23 28 29 30 30 30 30 30 30 30 30 30 30 30 30 30 30
 30 30 30 30 30 30  0  1  2  6  7  8 15 16 17 21 22 23 27 28 29 30 30 30
 30 30 30 30 30 30 30 30 30 30 30 30  0  1  2  3  6  7  8  9 14 15 16 17
 20 21 22 23 26 27 28 29 30 30 30 30 30 30 30 30 30 30  0  1  2  3  4  6
  7  8  9 10 13 14 15 16 17 19 20 21 22 23 25 26 27 28 29 30 30 30 30 30
  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29]

=== VALUE INFO ===
Constant_0_output              -> [4] (7)
Constant_1_output              -> [4] (7)
Constant_2_output              -> [4] (7)
Constant_3_output              -> [4] (7)
Slice_0_output                 -> [1, 10, 5, 5] (1)
ArgMax_0_reduced               -> [1, 1, 5, 5] (7)
Cast_0_output                  -> [1, 1, 5, 5] (2)
Constant_4_output              -> [4] (7)
Reshape_0_reshaped             -> [5, 1, 5, 1] (2)
Constant_5_output              -> [4] (7)
Expand_0_output                -> [5, 6, 5, 6] (2)
Constant_6_output              -> [1] (7)
Reshape_1_reshaped             -> [900] (2)
Constant_7_output              -> [72] (6)
Constant_8_output              -> [3] (7)
Constant_9_output              -> [3] (7)
Constant_10_output             -> [3] (7)
Constant_11_output             -> [3] (7)
Slice_1_output                 -> [1, 1, 1, 1] (1)
Cast_1_output                  -> [1, 1, 1, 1] (2)
Constant_12_output             -> [1] (7)
Reshape_2_reshaped             -> [1] (2)
Constant_13_output             -> [3] (7)
Constant_14_output             -> [3] (7)
Slice_2_output                 -> [1, 1, 1, 1] (1)
Cast_2_output                  -> [1, 1, 1, 1] (2)
Reshape_3_reshaped             -> [1] (2)
Mul_0_C                        -> [1] (2)
Add_0_C                        -> [1] (2)
Cast_3_output                  -> [1] (7)
Constant_15_output             -> [1] (7)
Mul_1_C                        -> [1] (7)
Add_1_C                        -> [1] (7)
Slice_3_output                 -> ['sym(unk__0)'] (6)
Constant_16_output             -> [24] (2)
ScatterElements_0_output       -> [900] (2)
Constant_17_output             -> [2] (7)
Reshape_4_reshaped             -> [30, 30] (2)
Constant_18_output             -> [4] (7)
Constant_19_output             -> [1] (2)
Pad_0_output                   -> [31, 31] (2)
Constant_20_output             -> [150] (6)
ReduceMax_0_reduced            -> [10] (1)
ReduceSum_0_reduced            -> [1] (1)
Cast_4_output                  -> [1] (2)
Constant_21_output             -> [1] (2)
Sub_0_C                        -> [1] (2)
Cast_5_output                  -> [1] (6)
Constant_22_output             -> [1] (6)
Sub_1_C                        -> [1] (6)
Constant_23_output             -> [1] (6)
Mul_2_C                        -> [1] (6)
Add_2_C                        -> [1] (6)
Slice_4_output                 -> ['sym(unk__1)'] (6)
Constant_24_output             -> [1] (7)
Reshape_5_reshaped             -> [30] (6)
Gather_0_output                -> [30, 31] (2)
Gather_1_output                -> [30, 30] (2)
Constant_25_output             -> [4] (7)
Reshape_6_reshaped             -> [1, 1, 30, 30] (2)
Constant_26_output             -> [1, 10, 1, 1] (2)
Equal_0_C                      -> [1, 10, 30, 30] (9)

=== NODE OUTPUT MAP ===
Constant_0_output              <- Constant
Constant_1_output              <- Constant
Constant_2_output              <- Constant
Constant_3_output              <- Constant
Slice_0_output                 <- Slice
ArgMax_0_reduced               <- ArgMax
Cast_0_output                  <- Cast
Constant_4_output              <- Constant
Reshape_0_reshaped             <- Reshape
Constant_5_output              <- Constant
Expand_0_output                <- Expand
Constant_6_output              <- Constant
Reshape_1_reshaped             <- Reshape
Constant_7_output              <- Constant
Constant_8_output              <- Constant
Constant_9_output              <- Constant
Constant_10_output             <- Constant
Constant_11_output             <- Constant
Slice_1_output                 <- Slice
Cast_1_output                  <- Cast
Constant_12_output             <- Constant
Reshape_2_reshaped             <- Reshape
Constant_13_output             <- Constant
Constant_14_output             <- Constant
Slice_2_output                 <- Slice
Cast_2_output                  <- Cast
Reshape_3_reshaped             <- Reshape
Mul_0_C                        <- Mul
Add_0_C                        <- Add
Cast_3_output                  <- Cast
Constant_15_output             <- Constant
Mul_1_C                        <- Mul
Add_1_C                        <- Add
Slice_3_output                 <- Slice
Constant_16_output             <- Constant
ScatterElements_0_output       <- ScatterElements
Constant_17_output             <- Constant
Reshape_4_reshaped             <- Reshape
Constant_18_output             <- Constant
Constant_19_output             <- Constant
Pad_0_output                   <- Pad
Constant_20_output             <- Constant
ReduceMax_0_reduced            <- ReduceMax
ReduceSum_0_reduced            <- ReduceSum
Cast_4_output                  <- Cast
Constant_21_output             <- Constant
Sub_0_C                        <- Sub
Cast_5_output                  <- Cast
Constant_22_output             <- Constant
Sub_1_C                        <- Sub
Constant_23_output             <- Constant
Mul_2_C                        <- Mul
Add_2_C                        <- Add
Slice_4_output                 <- Slice
Constant_24_output             <- Constant
Reshape_5_reshaped             <- Reshape
Gather_0_output                <- Gather
Gather_1_output                <- Gather
Constant_25_output             <- Constant
Reshape_6_reshaped             <- Reshape
Constant_26_output             <- Constant
Equal_0_C                      <- Equal
output                         <- Identity
task107.onnx.onnx, simplify ok=True
[ReshapeNode.shape_infer mismatch]
node: Reshape_6
input tensor: ['Gather_1_output', 'Constant_25_output']
output tensor: ['Reshape_6_reshaped']
srcshape: [0, 0] raw: 0
shape tensor: [ 1  1 30 30]
newshape: [1, 1, 30, 30] target: 900
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
[ReshapeNode.shape_infer mismatch]
node: Reshape_6
input tensor: ['Gather_1_output', 'Constant_25_output']
output tensor: ['Reshape_6_reshaped']
srcshape: [0, 0] raw: 0
shape tensor: [ 1  1 30 30]
newshape: [1, 1, 30, 30] target: 900
Name               Type               Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
-----------------  ---------------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
Expand_0           Expand                        0  0.00%            932  8.35%              4  9.52%       5x1x5x1    5x6x5x6
Add_1              Add                           1  0.01%             16  0.14%              1  2.38%       1          1
Sub_1              Sub                           1  0.01%              8  0.07%              1  2.38%       1          1
Mul_2              Mul                           1  0.01%              8  0.07%              1  2.38%       1          1
Add_2              Add                           1  0.01%              4  0.04%              0  0.00%       1          1
Sub_0              Sub                           1  0.01%              2  0.02%              1  2.38%       1          1
Equal_0            Equal                     9,000  97.11%         9,010  80.75%            10  23.81%      1x1x30x30  1x10x30x30
ArgMax_0           ArgMax                        0  0.00%            200  1.79%              0  0.00%       1x10x5x5   1x1x5x5
Mul_1              Mul                           1  0.01%              8  0.07%              0  0.00%       1          1
Mul_0              Mul                           1  0.01%              1  0.01%              0  0.00%       1          1
Add_0              Add                           1  0.01%              1  0.01%              0  0.00%       1          1
ReduceSum_0        ReduceSum                    10  0.11%              4  0.04%              0  0.00%       10         1
ReduceMax_0        ReduceMax                   250  2.70%             40  0.36%              0  0.00%       1x10x5x5   10
ScatterElements_0  ScatterElements               0  0.00%            924  8.28%             24  57.14%      900        900
Total              _                         9,268  100%          11,158  100%              42  100%        _          _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=    1000   11.29% shape=[1, 10, 5, 5]
ArgMax_0_reduced               int64      mem=     200    2.26% shape=[1, 1, 5, 5]
Cast_0_output                  uint8      mem=      25    0.28% shape=[1, 1, 5, 5]
Reshape_0_reshaped             uint8      mem=      25    0.28% shape=[5, 1, 5, 1]
Expand_0_output                uint8      mem=     900   10.16% shape=[5, 6, 5, 6]
Reshape_1_reshaped             uint8      mem=     900   10.16% shape=[900]
Slice_1_output                 float32    mem=       4    0.05% shape=[1, 1, 1, 1]
Cast_1_output                  uint8      mem=       1    0.01% shape=[1, 1, 1, 1]
Reshape_2_reshaped             uint8      mem=       1    0.01% shape=[1]
Slice_2_output                 float32    mem=       4    0.05% shape=[1, 1, 1, 1]
Cast_2_output                  uint8      mem=       1    0.01% shape=[1, 1, 1, 1]
Reshape_3_reshaped             uint8      mem=       1    0.01% shape=[1]
Mul_0_C                        uint8      mem=       1    0.01% shape=[1]
Add_0_C                        uint8      mem=       1    0.01% shape=[1]
Cast_3_output                  int64      mem=       8    0.09% shape=[1]
Mul_1_C                        int64      mem=       8    0.09% shape=[1]
Add_1_C                        int64      mem=       8    0.09% shape=[1]
Slice_3_output                 int32      mem=      96    1.08% shape=[24]
ScatterElements_0_output       uint8      mem=     900   10.16% shape=[900]
Reshape_4_reshaped             uint8      mem=     900   10.16% shape=[30, 30]
Pad_0_output                   uint8      mem=     961   10.85% shape=[31, 31]
ReduceMax_0_reduced            float32    mem=      40    0.45% shape=[10]
ReduceSum_0_reduced            float32    mem=       4    0.05% shape=[1]
Cast_4_output                  uint8      mem=       1    0.01% shape=[1]
Sub_0_C                        uint8      mem=       1    0.01% shape=[1]
Cast_5_output                  int32      mem=       4    0.05% shape=[1]
Sub_1_C                        int32      mem=       4    0.05% shape=[1]
Mul_2_C                        int32      mem=       4    0.05% shape=[1]
Add_2_C                        int32      mem=       4    0.05% shape=[1]
Slice_4_output                 int32      mem=     120    1.35% shape=[30]
Gather_0_output                uint8      mem=     930   10.50% shape=[30, 31]
Gather_1_output                uint8      mem=     900   10.16% shape=[30, 30]
Reshape_6_reshaped             uint8      mem=     900   10.16% shape=[1, 1, 30, 30]
output                         bool       mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=    8857 100.00%

It appears to require 9268 MACs + 8857 bytes + 315 params, yielding cost = 18440 and score = 15.178

[ReshapeNode.shape_infer mismatch]
node: Reshape_6
input tensor: ['Gather_1_output', 'Constant_25_output']
output tensor: ['Reshape_6_reshaped']
srcshape: [0, 0] raw: 0
shape tensor: [ 1  1 30 30]
newshape: [1, 1, 30, 30] target: 900
check: (9268, 8857, 315)


DONE !!!====================
len(examples) 266
wrong 0

Process finished with exit code 0



'''
