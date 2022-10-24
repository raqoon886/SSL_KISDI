import argparse
import os, posixpath
import re
import gc
import torch
from topic_transformer import *
from gensim.models.ldamodel import LdaModel
import warnings
warnings.filterwarnings("ignore")


def normalize_path(path : str):
    return path.replace(os.sep, posixpath.sep)


def normalize_text(text):
    doublespace_pattern = re.compile('\s+')
    text = str(text)
    text = doublespace_pattern.sub(' ', text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_model_path', type=str, default='', help='TopicTransformer model path')
    parser.add_argument('--lda_model_path', type=str, default='', help='LDA model path')

    args = parser.parse_args()
    transformer_model_path = normalize_path(args.transformer_model_path)
    lda_model_path = normalize_path(args.lda_model_path)

    while True:
        print("=============================\n")
        print("SCOPUS TOPIC INFERENCE MODULE\n")
        print("(E) Enter inference step")
        print("(Q) Quit\n")
        print("=============================")
        inp = input("type: ").strip()
        if inp not in ["E","Q","e","q"]:
            print("Not proper input\n")
            continue
        if inp in ["E","e"]:
            print("Entering inference process...")
            break
        if inp in ["Q","q"]:
            print("quit module")
            return

    model = ''
    while True:
        print("=============================\n")
        print("SCOPUS TOPIC INFERENCE MODULE\n")
        print("(L) Load TopicTransformer Model")
        print("(I) Start Inference")
        print("(Q) Quit\n")
        print("=============================\n")

        inp = input("type: ").strip()
        if inp not in ["L","I","Q","l","i","q"]:
            print("Not proper input")
            continue
        if inp in ["L","l"]:
            while True:
                print("=============================\n")
                print("Load TopicTransformer model\n")
                print("(M) Load TopicTransformer_MLP Model")
                print("(L) Load TopicTransformer_LSTM Model")
                print("(T) Load TopicTransformer_TEHead Model")
                print("(B) Back\n")
                print("=============================\n")

                inp = input("type: ").strip()
                if inp not in ["M","L","T","B","m","l","t","b"]:
                    print("Not proper input")
                    continue

                if inp in ["M","m"]:
                    model = TopicTransformer_MLP(output_dim=100,
                                                 transformer_model_name='xlm-roberta-base',
                                                 num_head_layers=3,
                                                 head_hidden_dims=[1024, 512, 256])
                    model.load_state_dict(torch.load(os.path.join(transformer_model_path, "TT_MLP_epoch60.pt")))
                    print("model loaded\n")
                    break

                if inp in ["L","l"]:
                    model = TopicTransformer_LSTM(output_dim=100,
                                                  transformer_model_name='xlm-roberta-base',
                                                  lstm_num_layers=2,
                                                  lstm_hidden_size=512)
                    model.load_state_dict(torch.load(os.path.join(transformer_model_path, "TT_LSTM_epoch20.pt")))
                    print("model loaded\n")
                    break

                if inp in ["T","t"]:
                    model = TopicTransformer_TEHead(output_dim=100,
                                                    transformer_model_name='xlm-roberta-base')
                    model.load_state_dict(torch.load(os.path.join(transformer_model_path, "TT_TEHead_epoch20.pt")))
                    print("model loaded\n")
                    break

                if inp in ["B","b"]:
                    break

        if inp in ["I","i"]:
            if model == '':
                print("Transformer model not loaded")
                continue
            else:
                break

        if inp in ["Q","q"]:
            print("quit module")
            return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ldamodel = LdaModel.load(lda_model_path)

    while True:
        print("=============================\n")
        print("Topic Inference Step\n")
        print("(I) Inference")
        print("(Q) Quit\n")
        print("=============================\n")

        inp = input("type: ").strip()
        if inp not in ["Q","I","q","i"]:
            print("Not proper input\n")
            continue

        if inp in ["Q","q"]:
            print("quit module")
            return

        if inp in ["I","i"]:
            while True:
                print("Enter Sentence: ")
                sentence = normalize_text(input().strip())

                model.eval()
                model.to(device)
                with torch.no_grad():
                    pred = model([sentence], device=device)

                pred = pred.to('cpu')[0].tolist()
                pred_dic = {i: a for i, a in enumerate(pred)}
                pred_sorted = sorted(pred_dic.items(), key=lambda x: x[1], reverse=True)
                for idx, (i, a) in enumerate(pred_sorted[:5]):
                    print("Top {} Topic Probability: {:.4f}".format(idx+1, a))
                    print(ldamodel.print_topic(i))

                print("End of Inference\n")
                torch.cuda.empty_cache()
                gc.collect()

                while True:
                    print("=============================\n")
                    print("(C) Continue")
                    print("(Q) Quit")
                    print("=============================\n")

                    inp = input("type: ").strip()
                    if inp not in ["C","c","Q","q"]:
                        print("Not proper input\n")
                        continue
                    if inp in ["Q", "q"]:
                        print("quit module")
                        return
                    if inp in ["C","c"]:
                        break

if __name__=='__main__':
    main()