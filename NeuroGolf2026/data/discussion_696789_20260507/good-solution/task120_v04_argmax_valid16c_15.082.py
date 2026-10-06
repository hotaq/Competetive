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
taskname = "task120"

# don't add doctsting
"""

TASK:
arc-gen task generator: task_50cb2852.py

ARC-Gen 50cb2852: input contains 2–4 non-overlapping solid rectangles
using colors 1, 2, 3. Output keeps rectangle borders unchanged and recolors
strict rectangle interiors to cyan 8.


ONNX SOLVER
Final strategy:
1. Work only on the known generator bound [0:16, 0:16].
   The generator width is small and height reaches at most 16, so the full
   30x30 canvas is unnecessary for detection.
2. Slice only channels 0..3 instead of all 10 channels, because input colors
   are restricted to background plus colors 1,2,3.
3. Decode this 4-channel crop with ArgMax into uint8 label16.
4. Derive foreground as label16 > 0.
5. Detect interiors by local 5-pixel foreground intersection:
      interior = min(center, left, right, up, down)
   using op.min([...]) as compact AND for 0/1 uint8 tensors.
6. Avoid full 16x16 validity ReduceSum. Since the valid ARC image is a
   continuous top-left rectangle, compute validity separably:
      top  = reduce_max(input[:,0:4,0:1,0:16], axis=1)
      left = reduce_max(input[:,0:4,0:16,0:1], axis=1)
      valid16 = left * top
   This reconstructs the rectangular valid mask cheaply.
7. Convert invalid cells to label 255 using the uint8 wrap trick:
      invalid255 = 0 - (1 - valid16)
   then label16 + invalid255 maps invalid padding to 255, which will not
   match any output color channel.
8. Overlay cyan using a single Where:
      if interior: 8 else label16_with_pad
9. Rebuild only 9 output channels on the 16x16 crop, because possible output
   labels are 0..3 and 8. Then pad channel axis and spatial axes to the final
   [1,10,30,30] output. This avoids the expensive full
   Equal(label, depth10) over [1,10,30,30].

Important NeuroGolf tricks used:
- Small crop before all real logic.
- Color-domain restriction: channels 0..3 only for input decoding.
- ArgMax has 0 MAC in this profiler, so it is cheaper than arithmetic decode.
- uint8 internal label/mask tensors reduce memory.
- op.min(list) is used as AND for 0/1 uint8 tensors.
- Separable top/left valid mask avoids full 4x16x16 reduction.
- 255 padding label prevents invalid padded cells from becoming background.
- Final one-hot is generated as [1,9,16,16] and padded to output, avoiding
  full 10-channel intermediate expansion.

Final measured result:
    wrong 0 / 266
    MACs   = 4224
    memory = 15968 bytes
    params = 92
    cost   = 20284
    score  = 15.082

Main remaining bottlenecks:
- crop [1,4,16,16] still costs 4096 bytes.
- ArgMax output int64 costs 2048 bytes before cast.
- 9-channel Equal/Cast on [1,9,16,16] costs 4608 bytes.
Further gains may require avoiding ArgMax or avoiding the 9-channel crop
one-hot rebuild, but this version is already a clean, correct, high-score
solution.
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
    # save your file as taskxxx_v01.py etc ...

    # v04 strategy:
    # 1) decode input once to a label map by ArgMax;
    # 2) detect rectangle interiors from bg channel only, but guard with
    #    valid16 = sum(onehot channels)==1 so invalid zero padding is not
    #    misread as foreground;
    # 3) cast label to uint8 and overlay cyan label 8 in uint8;
    # 4) final Equal(label_out, depth) is the graph output.

    zero_u8 = op_constant(0, np.uint8)
    one_u8 = op_constant(1, np.uint8)
    eight_u8 = op_constant(8, np.uint8)
    padding = op_constant(255, np.uint8)

    # Work only in the generator bound. Validate file has width up to 15,
    # random height can reach 16, so 16x16 is safe.
    crop10 = op_slice(input, [0, 0, 0, 0], [1, 4, 16, 16], [0, 1, 2, 3]) #only 4color

    label16 = op.arg_max(crop10, axis=1, keepdims=1)      # [1,1,16,16]
    label16 = op.cast(label16, to=np.uint8)


    # valid16 protects against the padded all-zero region outside the true ARC grid.
    # Without this guard, bg==0 in invalid padding looks like foreground and causes
    # cyan leakage on the right/bottom.

    ## [1,1,16,16]
    #valid16 = op.reduce_sum(crop10, op_constant([1], np.int64), keepdims=1) #1/0
    #use rank reduction trick for above
    top  = op_slice(input, [0, 0, 0, 0], [1, 4, 1, 16], [0, 1, 2, 3])
    left = op_slice(input, [0, 0, 0, 0], [1, 4, 16, 1], [0, 1, 2, 3])
    top  = op.cast(top, to=np.uint8)
    left = op.cast(left, to=np.uint8)
    top  = op.reduce_max(top, axes=[1], keepdims=1)
    left = op.reduce_max(left, axes=[1], keepdims=1)
    valid16 = op.mul(top,left)
    invalid16 = op.sub(one_u8, valid16)
    invalid16 = op.sub(zero_u8, invalid16) #-1=255 trick


    # Foreground is non-bg inside the valid image only.
    fg = op.cast(op.greater(label16, zero_u8), to=np.uint8)   # [1,1,16,16]

    # Axial neighbor maps:
    # left[y,x]  = fg[y,x-1], left edge padded 0
    # right[y,x] = fg[y,x+1], right edge padded 0
    # up[y,x]    = fg[y-1,x], top edge padded 0
    # down[y,x]  = fg[y+1,x], bottom edge padded 0
    fg_left = op.pad(
        op_slice(fg, [0, 0, 0, 0], [1, 1, 16, 15], [0, 1, 2, 3]),
        pads=op_constant([0, 0, 0, 1, 0, 0, 0, 0], np.int64),
        constant_value=zero_u8,
    )
    fg_right = op.pad(
        op_slice(fg, [0, 0, 0, 1], [1, 1, 16, 16], [0, 1, 2, 3]),
        pads=op_constant([0, 0, 0, 0, 0, 0, 0, 1], np.int64),
        constant_value=zero_u8,
    )
    fg_up = op.pad(
        op_slice(fg, [0, 0, 0, 0], [1, 1, 15, 16], [0, 1, 2, 3]),
        pads=op_constant([0, 0, 1, 0, 0, 0, 0, 0], np.int64),
        constant_value=zero_u8,
    )
    fg_down = op.pad(
        op_slice(fg, [0, 0, 1, 0], [1, 1, 16, 16], [0, 1, 2, 3]),
        pads=op_constant([0, 0, 0, 0, 0, 0, 1, 0], np.int64),
        constant_value=zero_u8,
    )

    # Interior iff center and all four axial neighbors are foreground.
    # Use op.min([...]) as compact AND for 0/1 uint8 tensors.
    interior16 = op.min([fg, fg_left, fg_right, fg_up, fg_down]) #list trick
    # interior16 = op.mul(op.mul(op.mul(op.mul(fg, fg_left), fg_right), fg_up), fg_down)

    # Pad small interior mask to full canvas.
    label16_with_pad = op.add(label16, invalid16)
    out16 = op.where(
        op.equal(interior16,zero_u8),
        label16_with_pad, #label16,
        eight_u8
    )


    #========== end of your code here ============================
    depth = op_constant(np.arange(9), np.uint8, [1, 9, 1, 1]) #up to cyan=8
    out16 = op.cast(
        op.equal(out16, depth),
        to=np.uint8,
    )

    output = op.pad(
        out16,
        pads=op_constant([0, 0, 0, 0, 0, 1, 14, 14], np.int64),
        #constant_value=padding,
    )  # uint8 [1,1,30,30]

    # Final one-hot. Keep this as graph output so [1,10,30,30] is not charged
    # as an intermediate by the Kaggle memory profiler.

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

        if (i < 2) or (wng):
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
            print("predict[0,0]")
            print_grid_color(predict[0,0])
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

task120.onnx.onnx, simplify ok=True
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
Name         Type         Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
-----------  ---------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
ReduceMax_0  ReduceMax              64  1.52%             16  0.25%              0  0.00%       1x4x1x16   1x1x1x16
Where_0      Where                   0  0.00%            257  3.99%              1  8.33%       1x1x16x16  1x1x16x16
Greater_0    Greater               256  6.06%            257  3.99%              1  8.33%       1x1x16x16  1x1x16x16
Add_0        Add                   256  6.06%            256  3.97%              0  0.00%       1x1x16x16  1x1x16x16
ArgMax_0     ArgMax                  0  0.00%          2,048  31.78%             0  0.00%       1x4x16x16  1x1x16x16
ReduceMax_1  ReduceMax              64  1.52%             16  0.25%              0  0.00%       1x4x16x1   1x1x16x1
Sub_0        Sub                   256  6.06%            257  3.99%              1  8.33%       1          1x1x16x16
Mul_0        Mul                   256  6.06%            256  3.97%              0  0.00%       1x1x1x16   1x1x16x16
Equal_0      Equal                 256  6.06%            256  3.97%              0  0.00%       1x1x16x16  1x1x16x16
Equal_1      Equal               2,304  54.55%         2,313  35.89%             9  75.00%      1x1x16x16  1x9x16x16
Sub_1        Sub                   256  6.06%            256  3.97%              0  0.00%       1          1x1x16x16
Min_0        Min                   256  6.06%            256  3.97%              0  0.00%       1x1x16x16  1x1x16x16
Total        _                   4,224  100%           6,444  100%              12  100%        _          _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=    4096   25.65% shape=[1, 4, 16, 16]
ArgMax_0_reduced               int64      mem=    2048   12.83% shape=[1, 1, 16, 16]
Cast_0_output                  uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Greater_0_C                    bool       mem=     256    1.60% shape=[1, 1, 16, 16]
Cast_1_output                  uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Slice_1_output                 uint8      mem=     240    1.50% shape=[1, 1, 16, 15]
Pad_0_output                   uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Slice_2_output                 uint8      mem=     240    1.50% shape=[1, 1, 16, 15]
Pad_1_output                   uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Slice_3_output                 uint8      mem=     240    1.50% shape=[1, 1, 15, 16]
Pad_2_output                   uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Slice_4_output                 uint8      mem=     240    1.50% shape=[1, 1, 15, 16]
Pad_3_output                   uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Min_0_min                      uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Equal_0_C                      bool       mem=     256    1.60% shape=[1, 1, 16, 16]
Slice_5_output                 float32    mem=     256    1.60% shape=[1, 4, 1, 16]
Cast_2_output                  uint8      mem=      64    0.40% shape=[1, 4, 1, 16]
ReduceMax_0_reduced            uint8      mem=      16    0.10% shape=[1, 1, 1, 16]
Slice_6_output                 float32    mem=     256    1.60% shape=[1, 4, 16, 1]
Cast_3_output                  uint8      mem=      64    0.40% shape=[1, 4, 16, 1]
ReduceMax_1_reduced            uint8      mem=      16    0.10% shape=[1, 1, 16, 1]
Mul_0_C                        uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Sub_0_C                        uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Sub_1_C                        uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Add_0_C                        uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Where_0_output                 uint8      mem=     256    1.60% shape=[1, 1, 16, 16]
Equal_1_C                      bool       mem=    2304   14.43% shape=[1, 9, 16, 16]
Cast_4_output                  uint8      mem=    2304   14.43% shape=[1, 9, 16, 16]
output                         uint8      mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=   15968 100.00%

It appears to require 4224 MACs + 15968 bytes + 92 params, yielding cost = 20284 and score = 15.082
check: (4224, 15968, 92)

DONE !!!====================
len(examples) 266
wrong 0


'''
