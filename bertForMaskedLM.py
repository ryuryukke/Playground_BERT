import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
"""
BERTはTransformerのEncoder部分のみを使っている。
クラス化してみた。
"""
# とりあえずpytorch-transfomerを触ってみよう
# Masked LMを試してみる(英語版)


class bertForMaskedLM():
    # default値がある変数は後ろに回さなければならない
    def __init__(self, text, masked_index, param_load_name="bert-base-uncased"):
        self.text = text
        self.param_load_name = param_load_name
        self.masked_index = masked_index
        self.tokenizer = BertTokenizer.from_pretrained(self.param_load_name)

    def get_masked(self, tokenized_text):
        tokenized_text[self.masked_index] = "[MASK]"
        return tokenized_text

    def tokenize(self):
        tokenized_text = self.tokenizer.tokenize(self.text)
        return tokenized_text

    def word_predict(self, tokens=[], segments=[]):
        model = BertForMaskedLM.from_pretrained(self.param_load_name)
        model.eval()
        tokens_tensor, segments_tensors = torch.tensor([tokens]), torch.tensor([segments])
        # If you have a GPU, put everything on cuda
        # tokens_tensor, segments_tensors = tokens_tensor.to("cuda"), segments_tensors.to("cuda")
        # model.to("cuda")
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]
        _, predicted_indexes = torch.topk(predictions[0, self.masked_index], k=5)
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
        return predicted_tokens


myModel = bertForMaskedLM("[CLS] I like to play football with my friends [SEP]", 5)


tokenized_text = myModel.tokenize()
masked_tokenized_text = myModel.get_masked(tokenized_text)
# 単語を対応するidに変換して、入力系列の完成
indexed_tokens = myModel.tokenizer.convert_tokens_to_ids(masked_tokenized_text)
# 今回、入力は文章は1文しか含まれないからすべてsegments_idsの要素は0
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
predicted_words = myModel.word_predict(indexed_tokens, segments_ids)
print(predicted_words)