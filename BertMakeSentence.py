"""
＜BERTのpre-trainedモデルを用いて文章生成＞
流れ：
1) Jumann(形態素解析器)を用いて、入力文章を形態素解析する
2) 京大の黒橋研究室が提供する「BERT日本語pre-trainedモデル」を使用
    ー　単語をidに変換して、テンソルに変換」
3) ある文章を用意して、先頭の単語に[MASK]を掛け予測をしたら、先頭の単語を予測結果に置き換え、
次の単語に[MASK]を掛け予測する、ということを繰り返すと入力文章に似た新たな文が生成ができるはずである。
"""


from pyknp import Juman
import torch
from pytorch_transformers import BertTokenizer, BertForMaskedLM, BertConfig


# MASKされた1単語を予測
def predict_one(tokens_tensor, mask_index):
    model.eval()
    # GPUを使用する場合
    # tokens_tensor = tokens_tensor.to("cuda")
    # model.to("cuda")
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
        _, predicted_indexes = torch.topk(predictions[0, mask_index], k=5)
        predicted_tokens = bert_tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
    # 予測された単語とIDを返す
    return predicted_tokens, predicted_indexes.tolist()


jumanpp = Juman()
# 入力文章
text = "ドッキリ"
# 形態素解析
result = jumanpp.analysis(text)
tokenized_text = [mrph.midasi for mrph in result.mrph_list()]
# [CLS]と[SEP]を加える
tokenized_text.insert(0, '[CLS]')
tokenized_text.append('[SEP]')
print(tokenized_text)
# 黒橋BERTモデル利用
config = BertConfig.from_json_file('/home/ryutokoike/Downloads/Japanese_L-12_H-768_A-12_E-30_BPE/bert_config.json')
model = BertForMaskedLM.from_pretrained("/home/ryutokoike/Downloads/Japanese_L-12_H-768_A-12_E-30_BPE/pytorch_model.bin", config=config)
# 日本語の辞書ファイルを読み込み
bert_tokenizer = BertTokenizer("/home/ryutokoike/Downloads/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt", do_lower_case=False, do_basic_tokenize=False)
# 単語をIDに変換し、テンソルに変換
indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

# 文生成
for i in range(1, len(tokens_tensor[0]) - 1):
    tmp = tokens_tensor
    # i番目を[MASK]に書き換える([MASK]のIDは4である)
    tmp[0, i] = 4
    predicted_tokens, predicted_indexes = predict_one(tmp, i)
    # 予測が[UNK]でない場合、tokens_tensorを予測で更新([UNK]のIDは1)
    # print(predicted_indexes)
    if predicted_indexes[0] != 1:
        tokens_tensor[0, i] = predicted_indexes[0]
    else:
        tokens_tensor[0, i] = predicted_indexes[1]
target_list = tokens_tensor.tolist()[0]
predict_list = bert_tokenizer.convert_ids_to_tokens(target_list)
predict_sentence = ''.join(predict_list[1:-1])
print('------ original_text -------')
print(text)
print('------ predict_text -------')
print(predict_sentence)