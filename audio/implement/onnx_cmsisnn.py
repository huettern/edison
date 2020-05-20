import onnx

model = onnx.load('.cache/ai_nemo/kws_id_mixed.onnx')

# graph = model.graph

for name, param in model.named_parameters():
  print(name, param.size())

# print(graph.input.pop())
# print(type(graph.node))
# print(graph.node.pop())
# print(graph.node.pop())
# print(graph.node.pop())
# print(graph.node.pop())
# print(graph.node.pop())
#
# print(type(graph))
# for node in graph:
#     print(node)
#
# print(model)
