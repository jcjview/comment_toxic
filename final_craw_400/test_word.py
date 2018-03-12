
train_word = {}
with open('train_word.txt', encoding="utf-8") as fp:
    for line in fp:
        v = line.strip().split("\t")
        if len(v) == 2:
            train_word[v[0]] = int(v[1])
print(len(train_word))

path = 'test_word.txt'
bad_dict = {}
with open(path, encoding="utf-8") as fp:
    for line in fp:
        v = line.strip().split("\t")
        if len(v) == 2:
            if v[0] in train_word:
                print(v[0])
            else:
                bad_dict[v[0]] = int(v[1])

sortedlist = sorted(bad_dict.items(), key=lambda d: d[1],reverse=True)

with open('sortword.txt', 'w', encoding="utf-8") as fp:
    for k, v in sortedlist:
        fp.write("%s\t%d\n" % (k, v))
