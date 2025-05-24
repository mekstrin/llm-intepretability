from nltk import word_tokenize


def split_expression(sentence):
    res = [word_tokenize(item.strip()) for item in sentence.split("[SEP]")]
    for i in range(len(res)):
        for j in range(len(res[i])):
            if res[i][j] == "''" or res[i][j] == "``":
                res[i][j] = '"'
    res_str = []
    for i, tokens in enumerate(res):
        res_str.extend(tokens)
        if i != len(res) - 1:
            res_str.append("[SEP]")
    return res_str
