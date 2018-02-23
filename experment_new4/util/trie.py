from collections import defaultdict


class Trie:
    """
    Implement a trie with insert, search, and startsWith methods.
    """
    def __init__(self):
        self.root = defaultdict()

    # @param {string} word
    # @return {void}
    # Inserts a word into the trie.
    def insert(self, word):
        current = self.root
        for letter in word:
            current = current.setdefault(letter, {})
        current.setdefault("_end")

    # @param {string} word
    # @return {boolean}
    # Returns if the word is in the trie.
    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current:
                return False
            current = current[letter]
        if "_end" in current:
            return True
        return False

    # @param {string} prefix
    # @return {boolean}
    # Returns if there is any word in the trie
    # that starts with the given prefix.
    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            if letter not in current:
                return False
            current = current[letter]
        return True

    # @param {string} text
    # @return {string}
    # Returns any word phrase in the trie
    def token(self,text):
        current = self.root
        ret=""
        for letter in text:
            if letter not in current:
                if  ret and self.search(ret):
                    return ret
                else:
                    return ""
            current = current[letter]
            ret+=letter
        if not ret and self.search(ret):
            return ret
        else:
            return ""

    def tokenize(self,text):
        ret=[]
        while text:
            t=self.token(text)
            if not t:
                text=text.split()
                text=text[1:]
                text=" ".join(text)
            else:
                ret.append(t)
                text=text.replace(t,'')
        return ret
# Now test the class

tree = Trie()
# test.insert('helloworld')
# test.insert('ilikeapple')
# test.insert('helloz')

# print (test.search('hello'))
# print (test.startsWith('hello'))
# print (test.search('ilikeapple'))
pre_dict={}
with open('rake.dat',encoding='utf-8') as file:
    for t in file:
        t=t.replace('[)(]','')
        ts=t.split(",")
        if len(ts)==2:
            text=ts[0]
            text=text.replace('(','')
            text = text.replace("'", '')
            pre_dict[text]=text.replace(" ","_")
            tree.insert(text)
print("trie len",len(pre_dict))

# text=' In my internet explorer is the formating of the page rather strange The text begins very late almost beneath the infobox Can this be changed so that the article looks better? Num CEST Num September Num Preceding WikipediaSignaturesuns'
# phrase = tree.tokenize(text)
# print(phrase)
# for p in phrase:
#     text = text.replace(p, pre_dict[p])
# print(text)
