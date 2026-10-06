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
taskname = "task119"

# don't add doctsting
"""
TASK
arc-gen task generator: task_508bd3b6.py
Given image image of red wall and initial pixels of cyan ray.
Draw the path of the ray and its rebound in green.
The whole scene may be flipped/rotated by gravity.

ONNX SOLVER

Task119 ONNX solution (NeuroGolf optimized, ~15.7 score)
Core idea:
----------
Reconstruct a V-shaped path from a partially visible cyan diagonal
by analytically extending the line and reflecting it at a wall.
The solution avoids simulation and instead uses a closed-form line
equation rendered efficiently via ScatterElements.

Algorithm overview:
------------------
1) Work on a fixed 12×12 grid (problem constraint).

2) Detect wall orientation:
   - Check red pixels along the middle row.
   - If wall is top/bottom, transpose grid to normalize into left/right case.

3) Detect wall position:
   - Find transition in red mask to determine wall coordinate (left/right).
   - Compute clipping bounds and sentinel index for red overwrite.

4) Detect diagonal direction:
   - Identify 2×2 cyan pattern:
       main (1 0 / 0 1) → slope +1
       anti (0 1 / 1 0) → slope -1
   - Extract one anchor point using TopK on weighted mask.

5) Analytical line reconstruction:
   - Forward ray:    x = tx + grad * (y - ty)
   - Rebound point:  y_hit = ty + grad * (wall_x - tx)
   - Backward ray:   x = wall_x - grad * (y - y_hit)

6) Efficient rendering:
   - Fuse forward and rebound rays into a single ScatterElements call.
   - Clip indices to valid region; use sentinel to restore wall color.

7) Restore visible cyan pixels after scatter.

8) If grid was transposed, transpose result back.

9) Convert label map to one-hot and pad to 30×30.

Key optimizations:
-----------------
- Geometry-first approach: operate on compact label/mask tensors.
- No full ArgMax over 10 channels (saves large memory).
- Single ScatterElements for both rays (fused update).
- Transpose normalization avoids dual renderer branches.
- Use uint8/bool tensors to minimize memory footprint.
- Avoid expensive GatherND and full-grid Where expansions.
- Use small 11×11 local detection for slope instead of global scans.

Performance:
------------
MACs:   ~2100–2300
Memory: ~8000–8500 bytes
Params: ~650
Score:  ~15.6–15.7

Notes:
------
- Final one-hot expansion (Equal) dominates MAC cost and is unavoidable.
- Scatter-based rendering is significantly cheaper than mask-based rendering.
- Transpose cost is small compared to avoiding duplicated logic.

Common pitfalls avoided:
-----------------------
- axis in ScatterElements must be constant (cannot be dynamic).
- op.max must take list: op.max([a,b]), not op.max(a,b).
- Avoid rank explosion from improper broadcasting.
- Avoid full 12×12 mask construction when scatter suffices.

Overall:
--------
A compact, high-efficiency solution using analytic geometry and
minimal ONNX graph operations, close to the practical performance ceiling.
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
    # v07: transpose-normalized renderer plus one fused two-ray ScatterElements call.
    # After normalization every case is treated as a left/right wall and uses
    # one x=f(y) ScatterElements renderer.  If the original wall was top/bottom,
    # transpose the rendered canvas back before output.

    def shape(s):
        return op_constant(s, np.int64)
    def c8u(v, shape_=None):
        return op_constant(v, np.uint8, shape_)
    def c32i(v, shape_=None):
        return op_constant(v, np.int32, shape_)
    def ci64(v, shape_=None):
        return op_constant(v, np.int64)
    def cast8u(v):
        return op.cast(v, to=np.uint8)
    def cast32i(v):
        return op.cast(v, to=np.int32)

    # -------- fixed 12x12 masks --------
    rd0 = op_slice(input, [0,2,0,0], [1,3,12,12], [0,1,2,3])         # red layer
    rd0 = cast8u(rd0)
    rd0 = op.reshape(rd0, shape([12,12]))                            # [12,12] 0/1

    cy0 = op_slice(input, [0,8,0,0], [1,9,12,12], [0,1,2,3])         # cyan layer
    cy0 = cast8u(cy0)
    cy0 = op.reshape(cy0, shape([12,12]))                            # [12,12] 0/1

    # Detect whether original wall is left/right using the middle row.
    # If false, the wall is top/bottom and we transpose masks so it becomes left/right.
    row_mid0 = op_slice(rd0, [5,0], [6,12], [0,1])
    row_mid0 = op.reshape(row_mid0, shape([12]))
    row_sum = op.reduce_sum(cast32i(row_mid0), axes=ci64([0]), keepdims=False)
    lr_exists = op.greater(row_sum, c32i(0))                         # scalar bool

    rd_t = op.transpose(rd0, perm=[1,0])                              # [12,12]
    cy_t = op.transpose(cy0, perm=[1,0])                              # [12,12]
    rd = op.where(lr_exists, rd0, rd_t)                               # normalized red
    cy = op.where(lr_exists, cy0, cy_t)                               # normalized cyan

    # Build original-label canvas in normalized coordinates.  No full ArgMax needed.
    lab12 = op.where(
        op.cast(rd, to=np.bool),
        c8u(2, [12,12]),
        op.where(op.cast(cy, to=np.bool), c8u(8, [12,12]), c8u(0, [12,12]))
    )

    # -------- detect normalized left/right wall and selected wall coordinate --------
    row = op_slice(rd, [5,0], [6,12], [0,1])
    row = op.reshape(row, shape([12]))                                # [12], uint8
    row0 = op_slice(row, [0], [11])
    row1 = op_slice(row, [1], [12])

    # right wall: 0 -> 1 transition. wall_x is last non-red col before wall.
    right_edge = cast8u(op.less(row0, row1))                          # [11]
    right_x = cast32i(op.arg_max(right_edge))                         # scalar

    # left wall: 1 -> 0 transition. wall_x is first non-red col after wall.
    left_edge = cast8u(op.greater(row0, row1))                        # [11]
    left_x0 = cast32i(op.arg_max(left_edge))                          # last red col
    left_x = op.add(left_x0, c32i(1))                                 # first non-red col

    left_exists = op.greater(op.reduce_sum(cast32i(left_edge), axes=ci64([0]), keepdims=False), c32i(0))
    wall_x = op.where(left_exists, left_x, right_x)                   # scalar

    # Scatter clipping parameters in normalized coordinates.
    right_limit = op.add(right_x, c32i(1))
    left_limit = op.sub(left_x, c32i(1))
    clip_lo_x = op.where(left_exists, left_limit, c32i(0))
    clip_hi_x = op.where(left_exists, c32i(11), right_limit)
    red_sentinel_x = op.where(left_exists, left_limit, right_limit)

    # -------- detect cyan 2-pixel diagonal pair and gradient in normalized grid --------
    # main: 1 0 / 0 1  => grad +1, anchor at (y,x)
    # anti: 0 1 / 1 0  => grad -1, anchor at top-right (y,x+1)
    cy00 = op_slice(cy, [0,0], [11,11], [0,1])
    cy11 = op_slice(cy, [1,1], [12,12], [0,1])
    cy01 = op_slice(cy, [0,1], [11,12], [0,1])
    cy10 = op_slice(cy, [1,0], [12,11], [0,1])
    main_pair = op.mul(cy00, cy11)                                   # [11,11]
    anti_pair = op.mul(cy01, cy10)                                   # [11,11]

    flat_ids = op_constant((np.arange(121, dtype=np.uint8) + 1).reshape(11,11), np.uint8)
    main_score = op.mul(main_pair, flat_ids)
    anti_score = op.mul(anti_pair, flat_ids)
    main_flat = cast32i(op.reshape(main_score, shape([121])))
    anti_flat = cast32i(op.reshape(anti_score, shape([121])))

    main_val, main_idx = op.top_k(main_flat, K=ci64(1))              # [1], [1]
    anti_val, anti_idx = op.top_k(anti_flat, K=ci64(1))              # [1], [1]
    use_main = op.greater(main_val, anti_val)                       # [1]

    mi = cast32i(main_idx)
    ai = cast32i(anti_idx)
    my = op.div(mi, c32i(11))
    mx = op.mod(mi, c32i(11))
    ay = op.div(ai, c32i(11))
    ax = op.mod(ai, c32i(11))

    tx_main = mx
    ty_main = my
    tx_anti = op.add(ax, c32i(1))
    ty_anti = ay

    tx = op.where(use_main, tx_main, tx_anti)                       # [1]
    ty = op.where(use_main, ty_main, ty_anti)                       # [1]
    grad = op.where(use_main, c32i(1, [1]), c32i(-1, [1]))           # [1]

    # -------- single normalized LR renderer: x = f(y), scatter along columns axis=1 --------
    yy = op_constant(np.arange(12, dtype=np.int32), np.int32)        # [12]
    dy = op.sub(yy, ty)
    xx_f = op.add(tx, op.mul(grad, dy))                              # [12]

    y_hit = op.add(ty, op.mul(grad, op.sub(wall_x, tx)))             # [1]
    xx_b = op.sub(wall_x, op.mul(grad, op.sub(yy, y_hit)))           # [12]

    # v07: render both forward and rebound rays with one ScatterElements call.
    # Build a [12,2] index tensor: two column updates per row.  This removes one
    # ScatterElements node plus one set of update/mask buffers while keeping the
    # same normalized x=f(y) geometry.
    canvas = lab12

    x_f_idx = op.reshape(op.clip(xx_f, clip_lo_x, clip_hi_x), shape([12,1]))
    x_b_idx = op.reshape(op.clip(xx_b, clip_lo_x, clip_hi_x), shape([12,1]))
    x_idx = op.concat([x_f_idx, x_b_idx], axis=1)                    # [12,2]

    upd = c8u(3, [12,2])                                             # green path
    upd = op.where(op.equal(x_idx, red_sentinel_x), c8u(2, [1]), upd)
    canvas = op.scatter_elements(
        canvas,
        indices=cast32i(x_idx),
        updates=upd,
        axis=1,
    )

    # Restore original cyan in normalized coordinates after scatter.
    canvas = op.where(op.cast(cy, to=np.bool), c8u(8, [12,12]), canvas)

    # If original was top/bottom, transpose rendered canvas back.
    canvas_back = op.transpose(canvas, perm=[1,0])
    canvas = op.where(lr_exists, canvas, canvas_back)

    # one-hot recover and pad to 30x30
    out12_lab = op.reshape(canvas, shape([1,1,12,12]))
    depth = op_constant(np.arange(10, dtype=np.uint8).reshape(1,10,1,1), np.uint8)
    out12 = op.equal(out12_lab, depth)
    output = op.pad(
        out12,
        pads=op_constant([0,0,0,0,0,0,18,18], np.int64),
    )

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

        if (i < 20) :#or (wng):
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
    if 1:
        for (i, input, predict, truth) in to_show:
            print("i", i, "--------------------------------")
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

task119.onnx.onnx, simplify ok=True
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
Name               Type               Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
-----------------  ---------------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
Where_4            Where                         0  0.00%              4  0.09%              0  0.00%       1          1
Where_5            Where                         0  0.00%             12  0.26%              2  0.33%       1          1
Transpose_1        Transpose                     0  0.00%            144  3.13%              0  0.00%       12x12      12x12
ReduceSum_0        ReduceSum                    12  0.56%              8  0.17%              1  0.17%       12         0
Where_2            Where                         0  0.00%            432  9.39%            288  47.52%      12x12      12x12
Mul_0              Mul                         121  5.67%            121  2.63%              0  0.00%       11x11      11x11
Clip_0             Clip                         24  1.13%             48  1.04%              0  0.00%       12         12
Mod_1              Mod                           1  0.05%              8  0.17%              1  0.17%       1          1
Greater_1          Greater                       1  0.05%              1  0.02%              0  0.00%       1          1
Where_13           Where                         0  0.00%            144  3.13%              0  0.00%       1          12x12
Mul_3              Mul                         121  5.67%            242  5.26%            121  19.97%      11x11      11x11
TopK_1             TopK                          0  0.00%             20  0.43%              1  0.17%       121        1
Where_10           Where                         0  0.00%              4  0.09%              0  0.00%       1          1
Greater_2          Greater                      11  0.52%             11  0.24%              0  0.00%       11         11
Add_1              Add                          12  0.56%             48  1.04%              0  0.00%       1          12
Add_3              Add                           1  0.05%              4  0.09%              0  0.00%       1          1
Where_12           Where                         0  0.00%            144  3.13%              0  0.00%       12x12      12x12
Where_8            Where                         0  0.00%              4  0.09%              0  0.00%       1          1
Sub_4              Sub                          12  0.56%             48  1.04%              0  0.00%       1          12
Mul_1              Mul                         121  5.67%            121  2.63%              0  0.00%       11x11      11x11
Mod_0              Mod                           1  0.05%              4  0.09%              0  0.00%       1          1
Sub_0              Sub                          12  0.56%             96  2.09%             12  1.98%       12         12
Add_4              Add                           1  0.05%              4  0.09%              0  0.00%       1          1
ArgMax_0           ArgMax                        0  0.00%              8  0.17%              0  0.00%       11         1
Sub_2              Sub                           1  0.05%              4  0.09%              0  0.00%       1          1
ArgMax_1           ArgMax                        0  0.00%              8  0.17%              0  0.00%       11         1
Mul_2              Mul                         121  5.67%            121  2.63%              0  0.00%       11x11      11x11
Add_2              Add                           1  0.05%              4  0.09%              0  0.00%       1          1
Add_0              Add                           1  0.05%              4  0.09%              0  0.00%       1          1
Transpose_2        Transpose                     0  0.00%            144  3.13%              0  0.00%       12x12      12x12
Where_3            Where                         0  0.00%            288  6.26%            144  23.76%      12x12      12x12
Where_0            Where                         0  0.00%            144  3.13%              0  0.00%       1          12x12
ReduceSum_1        ReduceSum                    11  0.52%              0  0.00%              0  0.00%       11         0
Where_9            Where                         0  0.00%              4  0.09%              0  0.00%       1          1
Sub_1              Sub                           1  0.05%              4  0.09%              0  0.00%       1          1
Where_7            Where                         0  0.00%              8  0.17%              1  0.17%       1          1
Greater_0          Greater                       1  0.05%              1  0.02%              0  0.00%       0          1
Less_0             Less                         11  0.52%             11  0.24%              0  0.00%       11         11
Equal_1            Equal                     1,440  67.51%         1,450  31.51%            10  1.65%       1x1x12x12  1x10x12x12
Mul_6              Mul                          12  0.56%             48  1.04%              0  0.00%       1          12
Div_1              Div                           4  0.19%              4  0.09%              0  0.00%       1          1
Transpose_0        Transpose                     0  0.00%            144  3.13%              0  0.00%       12x12      12x12
Sub_3              Sub                          12  0.56%             48  1.04%              0  0.00%       12         12
Mul_4              Mul                          12  0.56%             48  1.04%              0  0.00%       1          12
Greater_3          Greater                       1  0.05%              1  0.02%              0  0.00%       0          1
Where_1            Where                         0  0.00%            144  3.13%              0  0.00%       1          12x12
Div_0              Div                           4  0.19%              4  0.09%              0  0.00%       1          1
ScatterElements_0  ScatterElements               0  0.00%            144  3.13%              0  0.00%       12x12      12x12
Where_6            Where                         0  0.00%              4  0.09%              0  0.00%       1          1
Clip_1             Clip                         24  1.13%             48  1.04%              0  0.00%       12         12
Where_11           Where                         0  0.00%             49  1.06%             25  4.13%       12x2       12x2
Mul_5              Mul                           1  0.05%              4  0.09%              0  0.00%       1          1
Equal_0            Equal                        24  1.13%             24  0.52%              0  0.00%       12x2       12x2
TopK_0             TopK                          0  0.00%             12  0.26%              0  0.00%       121        1
Total              _                         2,133  100%           4,601  100%             606  100%        _          _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=     576    7.04% shape=[1, 1, 12, 12]
Cast_0_output                  uint8      mem=     144    1.76% shape=[1, 1, 12, 12]
Reshape_0_reshaped             uint8      mem=     144    1.76% shape=[12, 12]
Slice_1_output                 uint8      mem=      12    0.15% shape=[1, 12]
Reshape_1_reshaped             uint8      mem=      12    0.15% shape=[12]
Cast_1_output                  int32      mem=      48    0.59% shape=[12]
ReduceSum_0_reduced            int32      mem=       4    0.05% shape=[]
Greater_0_C                    bool       mem=       1    0.01% shape=[1]
Slice_2_output                 float32    mem=     576    7.04% shape=[1, 1, 12, 12]
Cast_2_output                  uint8      mem=     144    1.76% shape=[1, 1, 12, 12]
Reshape_2_reshaped             uint8      mem=     144    1.76% shape=[12, 12]
Transpose_0_transposed         uint8      mem=     144    1.76% shape=[12, 12]
Where_0_output                 uint8      mem=     144    1.76% shape=[12, 12]
Cast_3_output                  bool       mem=     144    1.76% shape=[12, 12]
Transpose_1_transposed         uint8      mem=     144    1.76% shape=[12, 12]
Where_1_output                 uint8      mem=     144    1.76% shape=[12, 12]
Cast_4_output                  bool       mem=     144    1.76% shape=[12, 12]
Where_2_output                 uint8      mem=     144    1.76% shape=[12, 12]
Where_3_output                 uint8      mem=     144    1.76% shape=[12, 12]
Slice_3_output                 uint8      mem=     121    1.48% shape=[11, 11]
Slice_4_output                 uint8      mem=     121    1.48% shape=[11, 11]
Mul_0_C                        uint8      mem=     121    1.48% shape=[11, 11]
Mul_1_C                        uint8      mem=     121    1.48% shape=[11, 11]
Reshape_3_reshaped             uint8      mem=     121    1.48% shape=[121]
Cast_6_output                  int32      mem=     484    5.91% shape=[121]
TopK_0_Values                  int32      mem=       4    0.05% shape=[1]
TopK_0_Indices                 int64      mem=       8    0.10% shape=[1]
Slice_5_output                 uint8      mem=     121    1.48% shape=[11, 11]
Slice_6_output                 uint8      mem=     121    1.48% shape=[11, 11]
Mul_2_C                        uint8      mem=     121    1.48% shape=[11, 11]
Mul_3_C                        uint8      mem=     121    1.48% shape=[11, 11]
Reshape_4_reshaped             uint8      mem=     121    1.48% shape=[121]
Cast_7_output                  int32      mem=     484    5.91% shape=[121]
TopK_1_Values                  int32      mem=       4    0.05% shape=[1]
TopK_1_Indices                 int64      mem=       8    0.10% shape=[1]
Greater_1_C                    bool       mem=       1    0.01% shape=[1]
Cast_8_output                  int32      mem=       4    0.05% shape=[1]
Mod_0_C                        int32      mem=       4    0.05% shape=[1]
Cast_9_output                  int32      mem=       4    0.05% shape=[1]
Mod_1_C                        int32      mem=       4    0.05% shape=[1]
Add_0_C                        int32      mem=       4    0.05% shape=[1]
Where_4_output                 int32      mem=       4    0.05% shape=[1]
Where_5_output                 int32      mem=       4    0.05% shape=[1]
Div_0_C                        int32      mem=       4    0.05% shape=[1]
Div_1_C                        int32      mem=       4    0.05% shape=[1]
Where_6_output                 int32      mem=       4    0.05% shape=[1]
Sub_0_C                        int32      mem=      48    0.59% shape=[12]
Mul_4_C                        int32      mem=      48    0.59% shape=[12]
Add_1_C                        int32      mem=      48    0.59% shape=[12]
Slice_7_output                 uint8      mem=      12    0.15% shape=[1, 12]
Reshape_5_reshaped             uint8      mem=      12    0.15% shape=[12]
Slice_8_output                 uint8      mem=      11    0.13% shape=[11]
Slice_9_output                 uint8      mem=      11    0.13% shape=[11]
Greater_2_C                    bool       mem=      11    0.13% shape=[11]
Cast_10_output                 uint8      mem=      11    0.13% shape=[11]
Cast_11_output                 int32      mem=      44    0.54% shape=[11]
ReduceSum_1_reduced            int32      mem=       4    0.05% shape=[]
Greater_3_C                    bool       mem=       1    0.01% shape=[1]
ArgMax_0_reduced               int64      mem=       8    0.10% shape=[1]
Cast_12_output                 int32      mem=       4    0.05% shape=[1]
Add_2_C                        int32      mem=       4    0.05% shape=[1]
Sub_1_C                        int32      mem=       4    0.05% shape=[1]
Where_7_output                 int32      mem=       4    0.05% shape=[1]
Less_0_C                       bool       mem=      11    0.13% shape=[11]
Cast_13_output                 uint8      mem=      11    0.13% shape=[11]
ArgMax_1_reduced               int64      mem=       8    0.10% shape=[1]
Cast_14_output                 int32      mem=       4    0.05% shape=[1]
Add_3_C                        int32      mem=       4    0.05% shape=[1]
Where_8_output                 int32      mem=       4    0.05% shape=[1]
Clip_0_output                  int32      mem=      48    0.59% shape=[12]
Reshape_6_reshaped             int32      mem=      48    0.59% shape=[12, 1]
Where_9_output                 int32      mem=       4    0.05% shape=[1]
Sub_2_C                        int32      mem=       4    0.05% shape=[1]
Mul_5_C                        int32      mem=       4    0.05% shape=[1]
Add_4_C                        int32      mem=       4    0.05% shape=[1]
Sub_3_C                        int32      mem=      48    0.59% shape=[12]
Mul_6_C                        int32      mem=      48    0.59% shape=[12]
Sub_4_C                        int32      mem=      48    0.59% shape=[12]
Clip_1_output                  int32      mem=      48    0.59% shape=[12]
Reshape_7_reshaped             int32      mem=      48    0.59% shape=[12, 1]
Concat_0_concat_result         int32      mem=      96    1.17% shape=[12, 2]
Where_10_output                int32      mem=       4    0.05% shape=[1]
Equal_0_C                      bool       mem=      24    0.29% shape=[12, 2]
Where_11_output                uint8      mem=      24    0.29% shape=[12, 2]
ScatterElements_0_output       uint8      mem=     144    1.76% shape=[12, 12]
Where_12_output                uint8      mem=     144    1.76% shape=[12, 12]
Transpose_2_transposed         uint8      mem=     144    1.76% shape=[12, 12]
Where_13_output                uint8      mem=     144    1.76% shape=[12, 12]
Reshape_8_reshaped             uint8      mem=     144    1.76% shape=[1, 1, 12, 12]
Equal_1_C                      bool       mem=    1440   17.59% shape=[1, 10, 12, 12]
output                         bool       mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=    8187 100.00%

It appears to require 2133 MACs + 8187 bytes + 663 params, yielding cost = 10983 and score = 15.696
check: (2133, 8187, 663)

DONE !!!====================
len(examples) 266
wrong 0


'''
