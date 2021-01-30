"""
Read in
https://towardsdatascience.com/simple-plagiarism-detection-in-python-2314ac3aee88

Source-code
https://gist.github.com/thomashikaru/cee1d5f4c198f5635036d67ff294fed4

"""
import re
import glob, os
from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import MLE, WittenBellInterpolated
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter


def train_texts(train_files, exclude, extension, n_ngram):
    # Training data file
    # train_data_file = "./train/treino.txt"

    # read training data
    #train_data_files = glob.glob('./train/*' + extension)
    train_data_files = train_files.copy()

    if (exclude):
        print("Arquivos no diretorio do treino antes de remover o item do test: ", train_data_files)
        train_data_files.remove(exclude)

    print("Arquivos utilizados no treino: ", train_data_files)

    train_texts = ''

    for train_data_file in train_data_files:

        try:
            #path_file_train =
            with open(os.path.join("./train", train_data_file), encoding='utf-8') as f:
                train_text = f.read().lower()
        except:
            print("Não foi possível acessar os arquivos de treino com a extensão ." + extension + " no diretório train.")

        # apply preprocessing (remove text inside square and curly brackets and rem punc)
        train_text = re.sub(r"\[.*\]|\{.*\}", "", train_text)
        train_text = re.sub(r'[^\w\s]', "", train_text)
        train_texts += train_text

    # pad the text and tokenize
    training_data = list(pad_sequence(word_tokenize(train_texts), n_ngram,
                                      pad_left=True,
                                      left_pad_symbol="<s>"))

    print("training_data", training_data)

    # generate ngrams
    ngrams = list(everygrams(training_data, max_len=n_ngram))
    print("Number of ngrams:", len(ngrams))

    # build ngram language models
    model = WittenBellInterpolated(n_ngram)
    model.fit([ngrams], vocabulary_text=training_data)
    print(model.vocab)

    return model

def test_text(model, extension, n_ngram, test_data_file, all_files):
    print("Arquivo utilizado no teste: ", test_data_file)

    print(test_data_file)
    if (all_files):
        path_file = os.path.join("./train",test_data_file)
    else:
        path_file = os.path.join("./test",test_data_file)
    # Read testing data
    with open(path_file, encoding='utf-8') as f:
        test_text = f.read().lower()

    test_text = re.sub(r'[^\w\s]', "", test_text)

    # Tokenize and pad the text
    testing_data = list(pad_sequence(word_tokenize(test_text), n_ngram,
                                     pad_left=True,
                                     left_pad_symbol="<s>"))
    print("Length of test data:", len(testing_data))
    print("testing_data", testing_data)

    # assign scores
    scores = []
    for i, item in enumerate(testing_data[n_ngram - 1:]):
        s = model.score(item, testing_data[i:i + n_ngram - 1])
        scores.append(s)

    scores_np = np.array(scores)

    # set width and height
    width = 8
    height = np.ceil(len(testing_data) / width).astype("int32")
    print("Width, Height:", width, ",", height)

    # copy scores to rectangular blank array
    a = np.zeros(width * height)
    a[:len(scores_np)] = scores_np
    diff = len(a) - len(scores_np)

    # apply gaussian smoothing for aesthetics
    a = gaussian_filter(a, sigma=1.0)

    # reshape to fit rectangle
    a = a.reshape(-1, width)

    # format labels
    labels = [" ".join(testing_data[i:i + width]) for i in range(n_ngram - 1, len(testing_data), width)]
    labels_individual = [x.split() for x in labels]
    labels_individual[-1] += [""] * diff
    labels = [f"{x:60.60}" for x in labels]

    # create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=a, x0=0, dx=1,
        y=labels, zmin=0, zmax=1,
        customdata=labels_individual,
        hovertemplate='%{customdata} <br><b>Score:%{z:.3f}<extra></extra>',
        colorscale="burg"))
    fig.update_layout({"height": height * 28, "width": 1000, "font": {"family": "Courier New"}})
    fig['layout']['yaxis']['autorange'] = "reversed"
    #fig.show()
    fig.write_html(file='./public/'+test_data_file+'.html')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define a extensão dos arquivos para o treino e teste
    extension = "md"
    # Define o numero de grams que serão adotados
    n_ngram = 4
    all_files = True

    list_files_in_directory_train = []

    # Lista dos arquivos do diretorio usado para o treino
    for train_data_file in os.listdir("./train"):
        if train_data_file.endswith(extension):
            list_files_in_directory_train.append(train_data_file)

    print("Arq no diretorio de treino: ",list_files_in_directory_train)


    # testing data files
    for test_data_file in os.listdir("./test"):
        # Se encontra um arquivo no diretorio de test com a extensão especificada
        if test_data_file.endswith(extension):
            all_files=False
            # treina com todos os arquivos do diretorio train
            model = train_texts(train_files=list_files_in_directory_train,
                                exclude=None, extension=extension, n_ngram=n_ngram)
            test_text(model=model, extension=extension, n_ngram=n_ngram,
                      test_data_file=test_data_file, all_files=all_files)

    if (all_files):
        for test_file in list_files_in_directory_train:
            # exclui um arquivo do treino e faz o teste com os demais
            model = train_texts(train_files=list_files_in_directory_train, exclude=test_file, extension=extension, n_ngram=n_ngram)
            test_text(model=model, extension=extension, n_ngram=n_ngram, test_data_file=test_file, all_files=True)