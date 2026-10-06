import numpy as np
import json
import onnx_tool
import math

from neurogolf_utils import *

#######################################################
#for dyanmic shape patching

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
            print(f"{name:30s}")
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

#######################################################

DATA_DIR  = '/media/hp/8TB-HDD/work/2026/kaggle/neurogolf/data/neurogolf-2026'
EXAMPLE_BAD_LIST = {
    ("task118", "train", 2), #violate generator rule
}


def load_examples(taskname='task001'):
    task_file = f'/{DATA_DIR}/{ taskname }.json'
    with open(task_file, 'r') as f:
        examples = json.load(f)
    return examples

def clean_examples(data, taskname='task001'):

    clean = {}
    for subset, exs in data.items():
        new_exs = []
        for idx, ex in enumerate(exs):
            # filter known bad case
            if (taskname, subset, idx) in EXAMPLE_BAD_LIST:
                print('skippling bad case:', taskname, subset, idx)
                continue

            inp = ex.get("input")
            H = len(inp)
            W = len(inp[0])

            # filter large grids
            if H > 30 or W > 30:
                print('larger than 30:', taskname, subset, idx)
                continue

            new_exs.append(ex)
        clean[subset] = new_exs

    return clean


def convert_color_to_onehot(g):
    H,W  = g.shape
    onehot= np.eye(10)[g]
    onehot = onehot.transpose((2,0,1))
    onehot = np.pad(onehot, ((0,0),(0,30-H),(0,30-W)))
    onehot = onehot[None,...].astype(np.float32)
    return onehot

def convert_color_to_onehot_hw(g):
    H, W = g.shape
    onehot = np.eye(10)[g]
    onehot = onehot.transpose((2, 0, 1))
    onehot = np.pad(onehot, ((0, 0), (0, 30 - H), (0, 30 - W)))
    onehot = onehot[None, ...].astype(np.float32)
    H=np.array([H], dtype=np.int64).reshape(1)
    W=np.array([W], dtype=np.int64).reshape(1)
    return onehot,H, W

def convert_onehot_to_color(onehot):
    t = onehot[0]
    t = t.sum(0)!=0
    y,x = np.where(t)
    H,W  = y.max()+1, x.max()+1
    g = onehot[0,:,:H,:W]
    g = g.argmax(0)
    return g


def print_grid_color(grid):
    COLORS = {
        i: f"\033[48;2;{r};{g};{b}m   \033[0m"
        for i, (r, g, b) in enumerate(
            [
                (0, 0, 0),  # black
                (30, 147, 255),  # blue
                (250, 61, 49),  # red
                (78, 204, 48),  # green
                (255, 221, 0),  # yellow
                (153, 153, 153),  # gray
                (229, 59, 163),  # magenta
                (255, 133, 28),  # orange
                (136, 216, 241),  # cyan
                (147, 17, 49),  # brown-ish
                (240, 240, 240),
                (146, 117, 86),
            ]
        )
    }
    for row in grid:
        print("".join(COLORS.get(x, "??") for x in row))
    print()


############################################################

def judge_onnx_profile_score(onnx_file):
  macs, memory, params = score_network(onnx_file)
  cost = macs + memory + params
  score = max(1.0, 25.0 - math.log(cost))

  print("Performance stats:")
  onnx_tool.model_profile(onnx_file)
  print()
  print(f"It appears to require {macs} MACs + {memory} bytes + {params} params, yielding cost = {cost} and score = {score:.3f}")
  print()
  return macs, memory, params, cost, score

####################################################################

def print_onnx_tool_profile(onnx_file):
    model = onnx_tool.loadmodel(onnx_file, {'verbose': False, 'constant_folding': True})
    g = model.graph
    g.graph_reorder_nodes()
    g.shape_infer(None)
    g.profile()
    print('## [ONNX-TOOL PROFILER FOR MACS and PARAMS]')
    onnx_tool.model_profile(onnx_file)
   # g.print_node_map()
    print()
    return int(sum(g.macs)), int(g.params)

def print_kaggle_memory_proile(filename):
    model = onnx.load(filename)
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph

    print('## [KAGGLE PROFILER FOR MEMORY]')

    init_names = {init.name for init in graph.initializer}
    io_names = {t.name for t in list(graph.input) + list(graph.output)}

    if io_names.intersection(init_names):
        print("Error: input/output name intersects initializer name.")
        return None

    if model.functions:
        print("Error: model contains functions.")
        return None

    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            print(f"Error: unsupported opset domain: {opset.domain}")
            return None

    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH,
                             onnx.AttributeProto.GRAPHS]:
                print("Error: model contains subgraph attribute.")
                return None

    rows = []
    total_mem = 0

    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not item.type.HasField("tensor_type"):
            continue

        name = item.name
        tensor_type = item.type.tensor_type

        if not tensor_type.HasField("shape"):
            print(f"Error: tensor {name} has no static shape.")
            return None

        shape = []
        num_elements = 1

        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                print(f"Error: tensor {name} has symbolic dim_param.")
                return None
            if not dim.HasField("dim_value"):
                print(f"Error: tensor {name} has missing dim_value.")
                return None

            shape.append(dim.dim_value)
            num_elements *= dim.dim_value

        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)

        if name in ["input", "output"]:
            mem = 0
        else:
            mem = num_elements * np.dtype(np_dtype).itemsize
            total_mem += mem

        rows.append((name, str(np_dtype), mem, shape))

    for name, dtype, mem, shape in rows:
        pct = f"{100 * mem / total_mem:.2f}%" if total_mem else "0%"
        print(f"{name:30s} {dtype:10s} mem={mem:8d} {pct:>8s} shape={shape}")

    print(f"{'Total':30s} {'_':10s} mem={total_mem:8d} 100.00%")

    return total_mem