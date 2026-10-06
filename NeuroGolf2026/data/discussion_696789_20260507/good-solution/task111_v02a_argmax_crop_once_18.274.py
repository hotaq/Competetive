
import math
import numpy as np
import onnx
import spox
from spox import argument, Tensor

# New versioned filename/name.
taskname = "task111"

"""
TASK
arc-gen task generator: task_48d8fb45.json
A gray marker points to a nearby 3-column object/pattern that should be cropped out.


ONNX SOLVER
Task111 – Closing Report (v02 ArgMax Dynamic Crop)
version: task111_v02a_argmax_crop_once.py

==================================================
1. Task Summary
==================================================
Given a 1x10x30x30 one-hot grid:
- A single marker exists in channel 5 within rows [0..6], cols [1..8]
- Extract a 3x3 crop around that marker:
    output = input[:, :, row+1:row+4, col-1:col+2]
- Place the crop at top-left of a 30x30 grid

==================================================
2. Key Optimization Idea
==================================================
Shift from:
    "enumerate all crops → select correct one"

to:
    "detect marker location → compute crop indices → crop once"

This eliminates:
- 56 candidate crops
- 56 pad operations
- 56 where operations

==================================================
3. Final Approach (v02)
==================================================
Pipeline:

1) Slice marker window:
    shape = [1,1,7,8]

2) Cast to uint8:
    reduces memory (float32 → uint8)

3) Flatten:
    [1,1,7,8] → [56]

4) ArgMax:
    idx ∈ [0..55]

5) Convert index → (row, col):
    row = idx // 8
    col = idx % 8

6) Compute crop:
    row_start = row + 1
    col_start = col

7) Dynamic Slice:
    shape = [1,10,3,3]

8) Pad once to 30x30

==================================================
4. ONNX Tricks / Hacks Used
==================================================

✓ ArgMax for coordinate detection
    - avoids ReduceSum + weight maps
    - removes large constant tensors

✓ uint8 marker representation
    - reduces intermediate buffer from 224 → 56 bytes

✓ Dynamic Slice (with shape patch)
    - avoids 56 static crops
    - requires manual shape patch:
    (avoid error"Error: tensor Slice_2_output has symbolic dim_param.")
        Slice_2_output = [1,10,3,3]

✓ Flatten + index math (Div/Mod)
    - replaces 2D coordinate maps

✓ Minimal intermediate tensors
    - no large broadcast tensors
    - no repeated Pad/Where chains

==================================================
5. Profiling Results
==================================================

MACs:
    0

Memory:
    808 bytes

Params:
    26

Total cost:
    834

Score:
    18.274

==================================================
6. Comparison vs Previous Versions
==================================================

Codex (baseline):
    many crops + many pads
    → very high cost
    score below 10

v01 (ReduceSum weights):
    224 MACs + 1136 memory
    score ≈ 17.686

v02 (ArgMax + uint8):
    0 MACs + 808 memory
    score ≈ 18.274

Main gains:
- removed 2 weight maps (56×2 floats)
- removed 2 Mul + 2 ReduceSum
- reduced dtype size
- reduced intermediate buffers

==================================================
7. Lessons Learned
==================================================

1. Always select indices, not tensors
   → avoid building large candidate sets

2. ArgMax is extremely powerful in ONNX
   → zero MAC cost
   → replaces weighted reductions

3. dtype matters (float32 vs uint8)
   → huge memory impact

4. Dynamic ops are good, but:
   → must patch shapes for Kaggle profiler

5. Avoid broadcast-heavy designs
   → they dominate memory cost

==================================================
8. Possible Further Improvements
==================================================

- Replace int64 ops with smaller dtype (if allowed)
- Remove unnecessary Unsqueeze/Concat chains
- Explore direct Gather-based cropping (may reduce Slice cost)
- Inline constants to reduce param count

==================================================
FINAL VERDICT
==================================================

This solution is:
✔ Correct
✔ Minimal
✔ Profiler-friendly
✔ Strong score (~18.27)

A clean and efficient final design.

"""


def make_onnx():
    SHAPE = (1, 10, 30, 30)
    input = argument(Tensor(np.float32, SHAPE))
    from spox.opset.ai.onnx import v17 as op

    def op_constant(value, dtype=np.int64, shape=None):
        value = np.asarray(value, dtype=dtype)
        if shape is not None:
            value = value.reshape(shape)
        return op.constant(value=value)

    def op_slice(x, starts, ends, axes, steps):
        return op.slice(
            x,
            starts=op_constant(starts),
            ends=op_constant(ends),
            axes=op_constant(axes),
            steps=op_constant(steps),
        )

    # ------------------------------------------------------------------
    # 1) Marker window.
    #    Original marker is channel 5, rows 0..6, cols 1..8.
    #    Shape before cast: 1 x 1 x 7 x 8, float32 one-hot values.
    #    Cast immediately to uint8 because we only need 0/1 for ArgMax.
    # ------------------------------------------------------------------
    marker_win_f = op_slice(input, [5,0,1], [6,7,9], [1,2,3], [1,1,1])
    marker_win_u8 = op.cast(marker_win_f, to=np.uint8)

    # Flatten 1x1x7x8 -> 56. With exactly one marker, ArgMax gives:
    #   idx = row_local * 8 + col_local
    flat = op.reshape(marker_win_u8, op_constant([56], dtype=np.int64))
    idx = op.arg_max(flat, axis=0, keepdims=0)

    # ------------------------------------------------------------------
    # 2) Convert idx -> crop start row/col.
    #    row_local = idx // 8
    #    col_local = idx % 8
    #    crop row_start = row_local + 1
    #    crop col_start = col_local
    # ------------------------------------------------------------------
    eight = op_constant(8, dtype=np.int64)
    three = op_constant(3, dtype=np.int64)
    one = op_constant(1, dtype=np.int64)

    row_local = op.div(idx, eight)
    col_local = op.mod(idx, eight)

    row_start_i = op.add(row_local, one)
    col_start_i = col_local
    row_end_i = op.add(row_start_i, three)
    col_end_i = op.add(col_start_i, three)

    # Slice starts/ends must be 1D tensors of length 2 for axes [2,3].
    unsq0 = op_constant([0], dtype=np.int64)
    row_start_1d = op.unsqueeze(row_start_i, axes=unsq0)
    col_start_1d = op.unsqueeze(col_start_i, axes=unsq0)
    row_end_1d = op.unsqueeze(row_end_i, axes=unsq0)
    col_end_1d = op.unsqueeze(col_end_i, axes=unsq0)

    crop_starts = op.concat([row_start_1d, col_start_1d], axis=0)
    crop_ends = op.concat([row_end_1d, col_end_1d], axis=0)

    # ------------------------------------------------------------------
    # 3) Crop once dynamically, then pad once to 30x30 top-left.
    # ------------------------------------------------------------------
    crop = op.slice(
        input,
        starts=crop_starts,
        ends=crop_ends,
        axes=op_constant([2, 3], dtype=np.int64),
        steps=op_constant([1, 1], dtype=np.int64),
    )

    output = op.pad(
        crop,
        pads=op_constant([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64),
        mode="constant",
    )

    model = spox.build(inputs={"input": input}, outputs={"output": output})

    # ------------------------------------------------------------------
    # Shape patch for Kaggle memory profiler.
    # Dynamic Slice starts/ends make ONNX shape inference emit symbolic dims.
    # The crop is always fixed size [1,10,3,3], so patch the value_info.
    #
    # Keep Slice_2_output for consistency with v01. Also patch any Slice output
    # that still has symbolic dims and rank 4 after build, except graph output.
    # ------------------------------------------------------------------
    def patch_vi(vi, shape, elem_type=onnx.TensorProto.FLOAT):
        tt = vi.type.tensor_type
        tt.elem_type = elem_type
        tt.shape.dim.clear()
        for d in shape:
            tt.shape.dim.add().dim_value = d

    target_shapes = {"Slice_1_output": [1, 10, 3, 3]}

    # If Spox naming changes, find the Slice feeding Pad/output by scanning nodes.
    for node in model.graph.node:
        if node.op_type == "Slice":
            for out in node.output:
                # The final crop Slice is the rank-4 float Slice before Pad. In this
                # small graph, patching any non-marker rank-unknown Slice output to
                # [1,10,3,3] by name fallback is safer after simplify may rename.
                if out.startswith("Slice_1"):
                    target_shapes[out] = [1, 10, 3, 3]

    found = set()
    for vi in list(model.graph.value_info) + list(model.graph.output):
        if vi.name in target_shapes:
            patch_vi(vi, target_shapes[vi.name])
            found.add(vi.name)

    for name, shape in target_shapes.items():
        if name not in found:
            model.graph.value_info.append(
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
            )

    onnx.checker.check_model(model)
    return model


######################################################################################
# Keep the same template/evaluation style as v01.
from onnxsim import simplify
import onnxruntime
from my_evalution import *


if __name__ == "__main__":
    onnx_file = f"{taskname}.onnx"

    model = make_onnx()
    model, ok = simplify(model)
    onnx.save(model, onnx_file)
    print(f"{onnx_file}, simplify ok={ok}")

    # --- check score profile -----
    macs, params = print_onnx_tool_profile(onnx_file)
    memory = print_kaggle_memory_proile(onnx_file)
    print()

    cost = macs + memory + params
    assert sum(score_network(onnx_file)) == cost
    score = max(1.0, 25.0 - math.log(max(1.0, cost)))
    print(
        f"It appears to require {macs} MACs + {memory} bytes + {params} params, "
        f"yielding cost = {cost} and score = {score:.3f}"
    )
    print()

    # --- check correctness -----
    session = onnxruntime.InferenceSession(model.SerializeToString())

    examples = load_examples(taskname)
    examples = clean_examples(examples, taskname)
    examples = examples["arc-gen"] + examples["train"] + examples["test"]

    wrong = 0
    for i in range(len(examples)):
        print("i", i, "--------------------------------")
        e = examples[i]

        inp = np.array(e["input"])
        inp = convert_color_to_onehot(inp)
        truth = np.array(e["output"])
        truth = convert_color_to_onehot(truth)

        (predict,) = session.run(["output"], {"input": inp})
        predict = predict.astype(np.float32)

        wng = 1 - np.array_equal(predict, truth)
        wrong += wng

        if (i < 1) or (wng):
            print("x")
            print_grid_color(convert_onehot_to_color(inp))
            print("p")
            print_grid_color(convert_onehot_to_color(predict))
            print("t")
            print_grid_color(convert_onehot_to_color(truth))

    print("DONE !!!====================")
    print("len(examples)", len(examples))
    print("wrong", wrong)

'''
numpy version:  2.4.4
onnx version:  1.21.0
onnxruntime version:  1.24.4
onnx_tool version:  1.0.1
NEUROGOLF_FILE_VERSION:  28-apr.01

task111.onnx, simplify ok=True
## [ONNX-TOOL PROFILER FOR MACS and PARAMS]
Name      Type      Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
--------  ------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
ArgMax_0  ArgMax               0  0.00%              0  0.00%              0  0.00%       56         0
Add_2     Add                  0  0.00%              0  0.00%              0  0.00%       0          0
Add_0     Add                  0  0.00%              0  0.00%              0  0.00%       0          0
Div_0     Div                  0  0.00%              0  0.00%              0  0.00%       0          0
Mod_0     Mod                  0  0.00%              0  0.00%              0  0.00%       0          0
Add_1     Add                  0  0.00%              0  0.00%              0  0.00%       0          0
Total     _                    0  100%               0  100%               0  100%        _          _

## [KAGGLE PROFILER FOR MEMORY]
input                          float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Slice_0_output                 float32    mem=     224   27.72% shape=[1, 1, 7, 8]
Cast_0_output                  uint8      mem=      56    6.93% shape=[1, 1, 7, 8]
Reshape_0_reshaped             uint8      mem=      56    6.93% shape=[56]
ArgMax_0_reduced               int64      mem=       8    0.99% shape=[]
Div_0_C                        int64      mem=       8    0.99% shape=[]
Add_0_C                        int64      mem=       8    0.99% shape=[]
Unsqueeze_0_expanded           int64      mem=       8    0.99% shape=[1]
Mod_0_C                        int64      mem=       8    0.99% shape=[]
Unsqueeze_1_expanded           int64      mem=       8    0.99% shape=[1]
Concat_0_concat_result         int64      mem=      16    1.98% shape=[2]
Add_1_C                        int64      mem=       8    0.99% shape=[]
Unsqueeze_2_expanded           int64      mem=       8    0.99% shape=[1]
Add_2_C                        int64      mem=       8    0.99% shape=[]
Unsqueeze_3_expanded           int64      mem=       8    0.99% shape=[1]
Concat_1_concat_result         int64      mem=      16    1.98% shape=[2]
Slice_1_output                 float32    mem=     360   44.55% shape=[1, 10, 3, 3]
output                         float32    mem=       0    0.00% shape=[1, 10, 30, 30]
Total                          _          mem=     808 100.00%

It appears to require 0 MACs + 808 bytes + 26 params, yielding cost = 834 and score = 18.274



'''