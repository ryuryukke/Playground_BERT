"""
マスク予測をpre-trainedモデルを使って試してみる
"""
# pytorch-transformersで日本語テキストを扱う
from pyknp import Juman
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

jumanpp = Juman()
text = "コーラにメントス入れてみたら大変なことになったwwww"
result = jumanpp.analysis(text)
tokenized_text = [mrph.midasi for mrph in result.mrph_list()]
tokenized_text.insert(0, '[CLS]')
tokenized_text.append('[SEP]')
print(tokenized_text)
masked_index = 1
tokenized_text[masked_index] = '[MASK]'
# masked_index_2 = 6
# tokenized_text[masked_index_2] = '[MASK]'

# 黒橋BERTモデル利用
config = BertConfig.from_json_file('/home/ryutokoike/Downloads/Japanese_L-12_H-768_A-12_E-30_BPE/bert_config.json')
model = BertForMaskedLM.from_pretrained("/home/ryutokoike/Downloads/Japanese_L-12_H-768_A-12_E-30_BPE/pytorch_model.bin", config=config)
# 日本語の辞書ファイルを読み込み
tokenizer = BertTokenizer("/home/ryutokoike/Downloads/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt", do_lower_case=False, do_basic_tokenize=False)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

_, predicted_indexes = torch.topk(predictions[0, masked_index], k=5)
# _, predicted_indexes_2 = torch.topk(predictions[0, masked_index_2], k=5)
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
# predicted_tokens_2 = tokenizer.convert_ids_to_tokens(predicted_indexes_2.tolist())
print(predicted_tokens)
# print(predicted_tokens_2)
# 仮にMASKが2つでもどちらも予想することができた
