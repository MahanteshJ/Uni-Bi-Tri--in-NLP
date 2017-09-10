import nltk
import math
import collections, nltk
from nltk.corpus import gutenberg
from nltk import bigrams
from nltk import trigrams
import numpy.random as npr
import numpy as np
import operator
import math
import random
def Creation_unigram(Firs):
    myFirst={}
    len_corpus=len(Firs)
    #print len_corpus
    for i in range(0,len(Firs)):
        #print (myFirst.get(Firs[i]))
        if myFirst.get(Firs[i]) >= (1.0/len_corpus):
            myFirst[Firs[i]]+=1.0/len_corpus
        #    print myFirst[First[i]]
        else :
            myFirst[Firs[i]]=1.0/len_corpus
         #   print myFirst[First[i]]

    return myFirst
def Creation_bigram(Firs,unigram,len_unigram):
    myFirst_bigram={}
    #print len_unigram
    len_corpus=len(Firs)
    for i in range(1,len(Firs)):
        if(Firs[i-1],Firs[i]) in myFirst_bigram:
            myFirst_bigram[Firs[i-1],Firs[i]]+=1.0/(unigram[Firs[i-1]]*len_unigram)
        else :
            myFirst_bigram[Firs[i-1],Firs[i]]=1.0/(unigram[Firs[i-1]]*len_unigram)
    return myFirst_bigram

def Creation_Trigram(Firs,bigram,unigram,len_unigram):
    myFirst_Trigram={}
    for i in range(2,len(Firs)):
         if (Firs[i-2],Firs[i-1],Firs[i]) in  myFirst_Trigram:
             myFirst_Trigram[Firs[i-2],Firs[i-1],Firs[i]]+=1.0/(bigram[Firs[i-2],Firs[i-1]]*(unigram[Firs[i-2]]*len_unigram))
             
         else:
             myFirst_Trigram[Firs[i-2],Firs[i-1],Firs[i]]=1.0/(bigram[Firs[i-2],Firs[i-1]]*(unigram[Firs[i-2]]*len_unigram))
    return myFirst_Trigram
def Trigram_manupulation(Firs,Old_Trigram,Old_bigram,Old_unigram):
    New_Trigram={}
    for i in range(2,len(Firs)):
        if (Firs[i-2],Firs[i-1],Firs[i]) not in New_Trigram:
            New_Trigram[Firs[i-2],Firs[i-1],Firs[i]]=Old_Trigram[Firs[i-2],Firs[i-1],Firs[i]]*Old_bigram[Firs[i-1],Firs[i]]*Old_unigram[Firs[i]]
           # print New_Trigram[Firs[i-2],Firs[i-1],Firs[i]] 
def Bigram(data,unigram,length):
    new_bigram={}
    for i in range(0,len(data)-1):
        if (data[i],data[i+1]) in new_bigram:
            new_bigram[data[i],data[i+1]]+=1.0/(unigram[data[i]]*length)
        else:
            new_bigram[data[i],data[i+1]]=1.0/(unigram[data[i]]*length)
    return new_bigram

def Cross_Entropy(first):
    Entropy=0.0
    for i in first:
        Entropy+=(1.0/len(first))*(-1*math.log(first[i],10))
        #print Entropy
    return Entropy
def Upper_Cross_Entropy(first):
    Entropy=0.0
    for i in first:
        Entropy+=first[i]*(-1*math.log(first[i],10))
        #print Entropy
    return Entropy
def calculting_prob(sentence,model):
    prob=0.0
    for i in range (0,len(sentence)):
        if sentence[i] not in  model:
            prob=0.0
            break
        else:
            prob+=math.log(model[sentence[i]],10)
            #prob*=model[sentence[+98*7/\8i]]
            
           # print sentence[i],model[sentence[i]],prob
    return prob
def calculting_prob_bi(sentence,model):
    prob=1.0
    #for i in range (1,len(sentence)):
    for i in model:
        #print i
        #print  sentence[[i-1],[i]]
        '''
        if sentence[[i-1],[i]] not in  model:
            prob=0.0
            break
        else:
            prob+=math.log(model[sentence[[i-1],[i]]],10)
            #prob*=model[sentence[i]]
            
           # print sentence[i],model[sentence[i]],prob
        '''
    return prob

def antilog(x):
   # return 10 ** x
    return pow(10,x)
print ("--First Corpus--")
First=nltk.corpus.gutenberg.words('whitman-leaves.txt')
First_unigram=Creation_unigram(First)
First_bigram=Creation_bigram(First,First_unigram,len(First))
New_bigram=Bigram(First,First_unigram,len(First))
First_Trigram=Creation_Trigram(First,First_bigram,First_unigram,len(First))
First_Entropy_uni=Cross_Entropy(First_unigram)
First_Upper_Entropy_uni=Upper_Cross_Entropy(First_unigram)
#print First_Upper_Entropy_uni,First_Entropy_uni
First_Entropy_bi=Cross_Entropy(First_bigram)
First_Upper_Entropy_bi=Upper_Cross_Entropy(First_bigram)
#print First_Upper_Entropy_bi,First_Entropy_bi
First_Entropy_tri=Cross_Entropy(First_Trigram)
First_Upper_Entropy_tri=Upper_Cross_Entropy(First_Trigram)

print ("--Second Corpus--")
Second=nltk.corpus.gutenberg.words('edgeworth-parents.txt')
Second_unigram=Creation_unigram(Second)
Second_bigram=Creation_bigram(Second,Second_unigram,len(Second))
Second_Trigram=Creation_Trigram(Second,Second_bigram,Second_unigram,len(Second))
Second_Entropy_uni=Cross_Entropy(Second_unigram)
Second_Upper_Entropy_uni=Upper_Cross_Entropy(Second_unigram)
#print Second_Entropy_uni,Second_Upper_Entropy_uni
Second_Upper_Entropy_bi=Upper_Cross_Entropy(Second_bigram)
Second_Entropy_bi=Cross_Entropy(Second_bigram)
#print Second_Entropy_bi,Second_Upper_Entropy_bi
Second_Entropy_tri=Cross_Entropy(Second_Trigram)
Second_Upper_Entropy_tri=Upper_Cross_Entropy(Second_Trigram)
#print Second_Entropy_tri,Second_Upper_Entropy_tri
#print Second_Entropy_uni,Second_Entropy_bi,Second_Entropy_tri

#for i in range(0,len(Second_unigram)):
 #       print Second_unigram[Second[i]]
print ("--Third Corpus--") 
Third=nltk.corpus.gutenberg.words('melville-moby_dick.txt')
Third_unigram=Creation_unigram(Third)
Third_bigram=Creation_bigram(Third,Third_unigram,len(Third))
Third_Trigram=Creation_Trigram(Third,Third_bigram,Third_unigram,len(Third))
Third_Entropy_uni=Cross_Entropy(Third_unigram)
Third_Entropy_bi=Cross_Entropy(Third_bigram)
Third_Entropy_tri=Cross_Entropy(Third_Trigram)

Test_sentence=nltk.corpus.gutenberg.words('C:\Users\Mahantesh\Desktop\NLP_ASS\Test.txt')
def authorship(a,b,c):
    sum=a+b+c;
    a1=100-(((a*1.0)/sum)*100)
    b1=100-(((b*1.0)/sum)*100)
    c1=100-(((c*1.0)/sum)*100)
    print ("AUTHORSHIP PROBABILITIES:",a1,b1,c1)
    
    if(a1>b1):
        if(a1>c1):
            print("")
            print ("TEST SENTENCE BELONGS TO CORPUS c1")
        elif(b1<c1):
            print("")
            print ("TEST SENTENCE BELONGS TO CORPUS c3")
            
    elif (b1>c1):
        print("")
        print ("TEST SENTENCE BELONGS TO CORPUS c2")
    else:
        print ("")
        print ("TEST SENTENCE BELONGS TO c3")
            
def test_create_unigram(Test,first):
    Entropy=0.0;
    for i in range(0,len(Test)):
        if Test[i]  in first :
            Entropy+=(1.0/len(Test))*(-1*math.log(first[Test[i]],10))
            #print Entropy
        else :
            Entropy+=(1.0/len(Test))*(-1*math.log(0.00000000005,10)) 
    return Entropy
def test_create_bigram(Test,first):
    Entropy=0.0
    for i in range(1,len(Test)):
        if  (Test[i-1],Test[i]) in first:
           Entropy+=(1.0/len(Test))*(-1*math.log(first[Test[i-1],Test[i]],10))
        else:
           Entropy+=(1.0/len(Test))*(-1*math.log(0.00000000005,10))
    return Entropy
def test_create_trigram(Test,first):
    Entropy=0.0
    for i in range(2,len(Test)):
        if (Test[i-2],Test[i-1],Test[i]) in first:
            Entropy+=(1.0/len(Test))*(-1*math.log(first[Test[i-2],Test[i-1],Test[i]],10))
        else:
           Entropy+=(1.0/len(Test))*(-1*math.log(0.00000000005,10))
    return Entropy
test_uni=test_create_unigram(Test_sentence,First_unigram);
test_uni1=test_create_unigram(Test_sentence,Second_unigram);
test_uni2=test_create_unigram(Test_sentence,Third_unigram);
test_bi=test_create_bigram(Test_sentence,First_bigram)
test_bi1=test_create_bigram(Test_sentence,Second_bigram)
test_bi2=test_create_bigram(Test_sentence,Third_bigram)
test_tri=test_create_trigram(Test_sentence,First_Trigram)
test_tri1=test_create_trigram(Test_sentence,Second_Trigram)
test_tri2=test_create_trigram(Test_sentence,Third_Trigram)
#test_bi=Creation_bigram(Test_sentence,test_uni,len(Test_sentence));
#test_tri=Creation_Trigram(Test_sentence,test_bi,test_uni,len(Test_sentence));
#Test_Entropy_uni=Cross_Entropy(test_uni)
#Test_Entropy_bi=Cross_Entropy(test_bi)
#Test_Entropy_tri=Cross_Entropy(test_tri)
print("--FOR CONSIDERED THREE CORPUS--")
print ("c1 ENTROPY--",First_Entropy_uni,First_Entropy_bi,First_Entropy_tri)
print ("c2 ENTROPY--",Second_Entropy_uni,Second_Entropy_bi,Second_Entropy_tri)
print ("c3 ENTROPY--",Third_Entropy_uni,Third_Entropy_bi,Third_Entropy_tri)
#print Test_Entropy_uni,Test_Entropy_bi,Test_Entropy_tri;
print("")
print ("--TEST SENTENCE--")
print ("UNIGRAM ENTROPY--","c1:",test_uni,"c2:",test_uni1,"c3",test_uni2)
print ("BIGRAM ENTROPY--","c1:",test_bi,"c2:",test_bi1,"c3:",test_bi2)
print ("TRIGRAM ENTROPY--","c1:",test_tri,"c2:",test_tri1,"c3:",test_tri2)
print("")
authorship(test_tri,test_tri1,test_tri2)
first=test_uni-First_Entropy_uni
second=test_uni1-Second_Entropy_uni
Third=test_uni2-Third_Entropy_uni





#here
sense = gutenberg.words('austen-sense.txt')
str1 = ' '.join(sense)
Utokens = nltk.word_tokenize(str1)
emma = gutenberg.words('austen-emma.txt')
str1 = ' '.join(emma)
Btokens = nltk.word_tokenize(str1)
#bi_tokens = bigrams(Btokens)
Bfdist=0
mel = gutenberg.words('melville-moby_dick.txt')
str1 = ' '.join(mel)
Ttokens = nltk.word_tokenize(str1)
tri_tokens = trigrams(Ttokens)
Tfdist=0
cacheb={}
cachet={}



def generate_model(val, num=15):
    val = sorted(val.items(), key = operator.itemgetter(1))
    sen=""
    for key,value in val:
        if(num<0):
            break;
        sen = sen+" "+key
        num -= 1
    print(sen)


def Biwords(words):
    for i in range(len(words) - 1):
            yield (words[i], words[i+1])

def Bdatabase():
        for w1, w2 in Biwords(Btokens):
            key = (w1)
            if key in cacheb:
                cacheb[key].append(w2)
            else:
                cacheb[key] = [w2]

def Bgenerate_markov_text(words, size=25):
        seed = random.randint(0, len(Btokens)-2)
        seed_word, next_word = Btokens[seed], Btokens[seed+1]
        w1, w2 = seed_word, next_word
        gen_words = []
        for i in range(size):
            gen_words.append(w1)
            w1, w2 = w2, random.choice(words[(w1)])
        gen_words.append(w2)
        return ' '.join(gen_words)


def triples(words):
    for i in range(len(words) - 2):
            yield (words[i], words[i+1], words[i+2])

def Tdatabase():
        for w1, w2, w3 in triples(Ttokens):
            key = (w1, w2)
            if key in cachet:
                cachet[key].append(w3)
            else:
                cachet[key] = [w3]

def Tgenerate_markov_text(words, size=25):
        seed = random.randint(0, len(Ttokens)-3)
        seed_word, next_word = Ttokens[seed], Ttokens[seed+1]
        w1, w2 = seed_word, next_word
        gen_words = []
        for i in range(size):
            gen_words.append(w1)
            w1, w2 = w2, random.choice(words[(w1, w2)])
        gen_words.append(w2)
        return ' '.join(gen_words)

#testset1 = "So fare thee well, poor devil of a Sub-Sub, whose commentator I am. Thou belongest to that hopeless, sallow tribe which no wine of this world will ever warm; and for whom even Pale Sherry would be too rosy-strong; but with whom one sometimes loves to sit, and feel poor-devilish, too; and grow convivial upon tears; and say to them bluntly, with full eyes and empty glasses,"
#testset1 = "humbled 8.41566660495178e-06 humiliating 8.41566660495178e-06 humiliations    8.41566660495178e-06 humility    3.36626664198071e-05 humour  7.5740999444566e-05 humoured    5.04939996297107e-05 humouring   8.41566660495178e-06 humsaid 8.41566660495178e-06 hundred 0.000176728998703987 hung    3.36626664198071e-05 hunt    8.41566660495178e-06 hunted  8.41566660495178e-06 hunters 3.36626664198071e-05 hunts   8.41566660495178e-06 hurried 9.25723326544696e-05 hurry   6.73253328396142e-05 hurrying    1.68313332099036e-05 hurt    8.41566660495178e-05 husband 0.000311379664383216 husbands    8.41566660495178e-05 hush    8.41566660495178e-06 hussy   8.41566660495178e-06 huswifes    8.41566660495178e-06 hysterical  1.68313332099036e-05 hysterics   4.20783330247589e-05 i   0.0161833268813223 ibe 8.41566660495178e-06 ice 8.41566660495178e-06 id  8.41566660495178e-06 idea    0.000193560331913891 ideai   8.41566660495178e-06 ideas   9.25723326544696e-05  dentify    8.41566660495178e-06 idle    5.89096662346624e-05 idled   8.41566660495178e-06 idleness    3.36626664198071e-05 idolized    8.41566660495178e-06 if  0.00244054331543602 ignorance   0.000100987999259421"
testset1 = " had done as sad a thing for herself as for them, and would have been a great deal happier if she had spent all the rest of her life at Hartfield. Emma smiled and chatted as cheerfully as she could, to keep him from such thoughts; but when tea came, it was impossible for him not to say exactly as he had said at dinner,"
model = Creation_unigram(Utokens)
print ("GENERATED TEXT:")
generate_model(model)
model = Creation_bigram(Utokens,model,len(Utokens))
#model=
#print (model)
Bdatabase()
print ("GENERATED TEXT:")
print(Bgenerate_markov_text(cacheb))

print ("-------------------------------------------------------------------")
Tdatabase()
print ("GENERATED TEXT:")
print(Tgenerate_markov_text(cachet))


