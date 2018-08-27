import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from util import MODEL_ROOT

g1_model_dir = os.path.join(MODEL_ROOT, 'graph_1/ckpt')
print_tensors_in_checkpoint_file(g1_model_dir,None,True,True)