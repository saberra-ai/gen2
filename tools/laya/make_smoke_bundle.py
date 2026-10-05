"""Create a tiny synthetic graph for integration tests, never a Laya substitute."""
import hashlib
import json
from pathlib import Path
import onnx
from onnx import helper as h, TensorProto as T
from tokenizers import Tokenizer, models, pre_tokenizers

root=Path("tests/fixtures/laya/smoke")
root.mkdir(parents=True,exist_ok=True)
inputs=[h.make_tensor_value_info(name,dtype,shape) for name,dtype,shape in [
    ("input_ids",T.INT64,["B","S"]),("attention_mask",T.INT64,["B","S"]),
    ("marker_pos",T.INT64,["B","K"]),("marker_mask",T.BOOL,["B","K"]),("qtype",T.INT64,["B"])]]
nodes=[h.make_node("Cast",["marker_pos"],["positions"],to=T.FLOAT),
    h.make_node("Where",["marker_mask","positions","negative"],["logits"]),
    h.make_node("Cast",["qtype"],["type_float"],to=T.FLOAT),
    h.make_node("Unsqueeze",["type_float","axes"],["column"]),
    h.make_node("Neg",["column"],["neg_column"]),
    h.make_node("Concat",["column","neg_column"],["act_logits"],axis=1)]
graph=h.make_graph(nodes,"GEN2_SYNTHETIC_TEST_ONLY",inputs,[h.make_tensor_value_info("logits",T.FLOAT,["B","K"]),h.make_tensor_value_info("act_logits",T.FLOAT,["B",2])],
    initializer=[h.make_tensor("negative",T.FLOAT,[],[-1e9]),h.make_tensor("axes",T.INT64,[1],[1])])
model=h.make_model(graph,opset_imports=[h.make_opsetid("",18)],ir_version=10)
onnx.checker.check_model(model)
onnx.save(model,str(root/"model.onnx"))
vocab={t:i for i,t in enumerate(["[PAD]","[CLS]","[SEP]","[MASK]","[UNK]","a","b","false","true","low","high","level","0","1"])}
tok=Tokenizer(models.WordLevel(vocab,unk_token="[UNK]"));tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(root/"tokenizer.json"))
(root/"config.json").write_text('{"temperature":[1,1,1]}\n')
(root/"LICENSE.txt").write_text("Synthetic test graph and tokenizer created for gen2, MIT license. No model weights.\n")
files={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.iterdir() if p.name!="manifest.json"}
manifest=dict(schema="gen2-laya-bundle/v1",model_repository="convaiinnovations/laya",model_revision="7b928d828b7b0e022f929d9bd2e44165aa270148",code_revision="8a6e1328cce2460a0e5aa348ad465bb1b5821cd2",checkpoint="english",graph="model.onnx",tokenizer="tokenizer.json",config="config.json",calibration=None,licenses=["LICENSE.txt"],dtype="fp32",opset=18,exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions={"fixture":"SYNTHETIC_TEST_ONLY"},envelope=dict(max_len=512,head_max_len=192,max_batch=2,max_options=16,reservation_mb=1),special_tokens=dict(cls=1,sep=2,pad=0,mask=3,mask_text="[MASK]"),files=files)
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
