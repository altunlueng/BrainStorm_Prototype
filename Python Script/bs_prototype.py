# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 13:35:12 2018

@author: Engin Altunlu
"""

# Brain Storm Prototype
# Trials with wikipedia articles



# retrieval form wikipedia
import wikipedia 
from mediawiki import MediaWiki

import wikipediaapi
####
import numpy as np # mathematical stuff
import matplotlib.pyplot as plt # it's usually for plotting
import pandas as pd #dataset processing lib
# counter
from collections import Counter as cnt
# text mining libs
# Need to remove stop words + stemming 
#needed libraries:
import re
# nltk library helps us to rmove the words that we call it stop words like 'this' 'it' etc.
import nltk
# we download the stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english") # i will use it in the function
# stemming library
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()  

# here there is a trick that in a sentence there can be some dots which are not the end of a phrase
# so on nltk library there ara some corpus to train and construct a robust model to split the sentence
# Training
    
#we train our stuff to extract the sentences
# ******* Later, for example we can train our stuff with the first article we picked******
from nltk.corpus import gutenberg
#print( dir(gutenberg))
#print (gutenberg.fileids())
 
text = ""
for file_id in gutenberg.fileids():
    text += gutenberg.raw(file_id)
 
#print(len(text))

# libraries needed
from pprint import pprint
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
#training
trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
trainer.train(text)
tokenizer = PunktSentenceTokenizer(trainer.get_params())

import os
os.chdir('D:/Brain Storm/Data')

#################"
"""
# some tests
wikipedia = MediaWiki() #creating wikipedia object
p = wikipedia.page('Chess') #retrieving the page

# getting the categoriy tree
# sorts a list
# wann use this as 'class column' when clustering
cats = p.categories 
cats[0] # main category
"""

#########
# wikipedia-api

# initializing the objects
wikipedia = MediaWiki() #creating wikipedia object

wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)


# FUNCS #############
# getting all pages from a category. returns list of titles
def get_categorymembers(categorymembers, level=0, max_level=2):
        t = []
        for c in categorymembers.values():
            #print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
            t.append(c.title)
            """if c.ns == wikipediaapi.Namespace.CATEGORY and level <= max_level:
                get_categorymembers(c.categorymembers, level + 1)"""
        return t


# getting contexts of pages
def sort_textset(p_titles):
    A = [] # all information : id, title, text, category
    i = 1
    for p_name in p_titles:
        t = []
        t.append(i)
        t.append(p_name)
        p = wiki.page(p_name)
        #pm = wikipedia.page(p_name)
        c = p.categories
        t.append(p.text)
        t.append(next (iter (c.keys())))
        A.append(t)
        i= i+1
    return A

# it will return the word frequency in the text
# need to call stemmer before this
def word_count(text,word):
    t = text.split()
    count = 0
    #ps = PorterStemmer()
    for w in t:
        if ps.stem(w) == ps.stem(word):
            count = count+1
    return count

#- we count how many times the other words appeared with the primary word
def count_companions(contents,word):
    #ps = PorterStemmer()
    counts = dict()
    for text in contents:
        for w in text:
            if (ps.stem(w) is not ps.stem(word)):
                counts[w] = counts.get(w, 0) + 1
    return counts
   
# articles is the dataset that i have, which includes many articles from wikipedia
# so it takes the 'text' column's member; a is the index of that article
# i pick for now 0
# articles a dataframe
def get_p_words(articles,a):
    # Getting Primary words
    p_words = [] # Primary words

    content= re.sub('[^a-zA-Z]', ' ', str(articles['Text'][a])) 
    # we make everything lower case
    content= content.lower()
    # we split the texts here
    content_copy = content.split() # to get the real words- not stemmed
    content = content.split()
    # set function makes it faster because it takes it like an input
    # so it is more useful  for long articles
    content = [ps.stem(word) for word in content if not word in set(cachedStopWords)]
    # we join them again into one context but stemmed and cleaned
    # i don't want to do that 
    #content = ' '.join(content)
    #i want to get the real words so i compare with the copy one
    for word in content_copy:
        if ps.stem(word) in content:
            if word not in p_words:
                p_words.append(word) #avoiding repeats
    return p_words     


           
# we get the articles which includes primary words from our root artcile
# p_words = primary words - list, k= how many articles we wanna sort
def look_Pwords(p_words,k=5):
    all_articles =[]
    for word in p_words:
        print('Loking for word: ' + word)
        w_count_list = [] # to get the maximum used article
        pageList = wikipedia.search(word)
        for p in pageList:
            page = wiki.page(p)
            page_text = page.text
            count = word_count(page_text,word)
            w_count_list.append(count)
        if len(pageList) <= k:
            t = sort_textset(pageList)
            for e in t:
                all_articles.append(e)
        else:
            max_list = sorted(zip(w_count_list, pageList), reverse=True)[:k] #mximum counted articles title
            max_pages = []
            for l in max_list:
                max_pages.append(l[1])
            t = sort_textset(max_pages)
            for e in t:
                all_articles.append(e)  
    return all_articles


# getting sentences with the primary word
def get_sentences(word,sentences):
    #ps = PorterStemmer()
    #print(word)
    #print(sentences)
    wanted_sentences = []
    for se in sentences:
        #print(se)
        splitted = se.split()
        #print(s)
        for s in splitted:
            if ps.stem(word) in ps.stem(s):
                wanted_sentences.append(se)
    #print(wanted_sentences)
    return wanted_sentences


# we get rid of everything here
# numbers etc.
# lower the case and stemming
def remove_nonMeaningful(text):
    # get rid of numbers
    content= re.sub('[^a-zA-Z]', ' ', str(text)) 
    # we make everything lower case
    content= content.lower()
    # we split the phrases here
    content = content.split()
    # set function makes it faster because it takes it like an input
    # so it is more useful  for long articles
    content = [ps.stem(word) for word in content if not word in set(cachedStopWords)]
    return content
  

#3,4 and 5. steps of the algo
# ***** IMPORTANT*** it gets data frame as input (because of excel thing): to change later on
def linking_words(p_words,all_articles,tokenizer):
    all_companions = {} # it will be a dict of dicts
    for j in range(0,len(all_articles)):
        article = all_articles['Text'][j]
        all_sentences = tokenizer.tokenize(article)
        contents = []
        for i in range(0,len(p_words)):
            # i wanna make first column with primary word and second column with his dict : companion counts
            word = p_words['Primary_Word'][i]
            print( str(j) +"th article with title : -"+ str(all_articles['Title'][j]) + "- for the word : " + str(word) + "...")
            #print(article)
            sentences = get_sentences(word,all_sentences) # getting the sentenceswith the primary word
            #print(sentences)
            for se in sentences:
                contents.append(remove_nonMeaningful(se))
            if word not in all_companions:
                companions = count_companions(contents,word) # returns a dict with key : campanion word and its number
                all_companions[word] = companions # dict of p_word and its companions as dict (dict in a dict)
            else:
                companions = all_companions[word]
                temp_companions = count_companions(contents,word)
                for w in temp_companions.keys():
                    companions[w] = companions.get(w, 0) + 1
                all_companions[word] = companions
    return all_companions

# to count and get the percentage of a cluster with its largest known group
# p = max(group)/cluster_member_number
# arranged is a list of list which contains words and their clusters
# k is the cluster number
def purity(arranged,k,groups,pwords):
    purities = []
    for cluster in range(0,k):
        member_num = 0
        group_dict = {}
        for c in arranged:
            word = c[0]
            if c[1] == cluster:
                member_num = member_num +1
                for i in range(0,len(pwords)):
                    if word == pwords[i]:
                        group_name = str(groups[i])
                        group_dict[group_name] = group_dict.get(group_name,0) +1
        
        
        counted = cnt(group_dict)
        t = counted.most_common(1)
        max_group = t[0][1]
        p = max_group/member_num
        purities.append(p)
    total = 0
    for p in purities:
        total = total + p
    purity = total/4
    return purity

######  ########
"""  
# Crawling #
# getting first category to analyze  
cat = wiki.page("Category:Acoustics")
#print("Category members: Category:Acoustics")
# page titles list
members = get_categorymembers(cat.categorymembers)
main_set = sort_textset(members)

#getting other categories we think that they can be related**
cat = wiki.page("Category:Architecture")
#print("Category members: Category:Acoustics")
# page titles list
members = get_categorymembers(cat.categorymembers)
rel_set = sort_textset(members)

#
cat = wiki.page("Category:Sound")
#print("Category members: Category:Acoustics")
# page titles list
members = get_categorymembers(cat.categorymembers)
rel_set_2 = sort_textset(members)

# Not related - less related
cat = wiki.page("Category:Biology")
#print("Category members: Category:Acoustics")
# page titles list
members = get_categorymembers(cat.categorymembers)
not_rel_set = sort_textset(members)
##############

### Writing to excel

from pandas import ExcelWriter
writer = ExcelWriter('WikiAcoustics.xlsx')
df = pd.DataFrame(main_set, columns = ['ID','Title','Text','Category'])
df.to_excel(writer, 'Acoustics') 
writer.save()   

from pandas import ExcelWriter
writer = ExcelWriter('WikiArchitecture.xlsx')
df = pd.DataFrame(rel_set, columns = ['ID','Title','Text','Category'])
df.to_excel(writer, 'Architecture') 
writer.save()   

from pandas import ExcelWriter
writer = ExcelWriter('WikiSound.xlsx')
df = pd.DataFrame(rel_set_2, columns = ['ID','Title','Text','Category'])
df.to_excel(writer, 'Sound') 
writer.save()   

from pandas import ExcelWriter
writer = ExcelWriter('WikiBiology.xlsx')
df = pd.DataFrame(not_rel_set, columns = ['ID','Title','Text','Category'])
df.to_excel(writer, 'Biology') 
writer.save()   
"""
############################################
# you can get them from their xslx later on
from pandas import ExcelFile
xl= pd.ExcelFile('WikiAcoustics.xlsx')
main_set = xl.parse('Acoustics')
#
xl= pd.ExcelFile('WikiArchitecture.xlsx')
rel_set = xl.parse('Architecture')
#
xl= pd.ExcelFile('WikiSound.xlsx')
rel_set_2 = xl.parse('Sound')
#
xl= pd.ExcelFile('WikiBiology.xlsx')
not_rel_set = xl.parse('Biology')
####
# I wanna create a full dataset with all groups
dataset = main_set
dataset = main_set.append(rel_set,ignore_index=True)
a = dataset.append(rel_set_2,ignore_index=True)
dataset = a
a = dataset.append(not_rel_set,ignore_index=True)
dataset = a
#del dataset['ID'] # i remove ID
# iwrite the new ids here
for i in range(1, len(dataset)):
    dataset['ID'][i] = i
#####


# I'll get just one article and sort the primary words.
# Not all the articles for now

p_words = get_p_words(dataset,0)

xl= pd.ExcelFile('P_words_selection.xlsx')
p_w_s = xl.parse('Primary_Word') # data frame

# data frame to a list
p_words_selected =[]
groups = []
for i in range (0,len(p_w_s)):
    word = p_w_s['Primary_Word'][i]
    group = p_w_s['Group'][i]
    p_words_selected.append(word)
    groups.append(group)
# Primary words research
# getting all articles including primary word. It takes a list

#all_articles = look_Pwords(p_words,5)
all_articles_selected = look_Pwords(p_words_selected,5)


######## For further work
# I write it to excel
from pandas import ExcelWriter
writer = ExcelWriter('P_Words_Articles_From_NoiseReduction_Coefficient.xlsx')
df = pd.DataFrame(all_articles, columns = ['ID','Title','Text','Category'])
df.to_excel(writer, 'P_Words_Articles') 
writer.save()  
# i write also the primary words that i extracted
#
from pandas import ExcelWriter
writer = ExcelWriter('P_W_Selected_Articles_From_NoiseReduction_Coefficient.xlsx')
df = pd.DataFrame(all_articles_selected, columns = ['ID','Title','Text','Category'])
df.to_excel(writer, 'P_Words_Articles') 
writer.save()  
from pandas import ExcelWriter
writer = ExcelWriter('Primaries_NoiseReduction_Coefficient.xlsx')
df = pd.DataFrame(p_words, columns = ['Primary_Word'])
df.to_excel(writer, 'Primary_Words') 
writer.save()  
########
"""
from pandas import ExcelFile
xl= pd.ExcelFile('P_Words_Articles_From_NoiseReduction_Coefficient.xlsx')
all_articles = xl.parse('P_Words_Articles') # data frame
"""
#
from pandas import ExcelFile
xl= pd.ExcelFile('P_W_Selected_Articles_From_NoiseReduction_Coefficient.xlsx')
all_articles_selected = xl.parse('P_Words_Articles') # data frame
"""
xl= pd.ExcelFile('Primaries_NoiseReduction_Coefficient.xlsx')
p_words = xl.parse('Primary_Words') # data frame
"""
xl= pd.ExcelFile('P_words_selection.xlsx')
p_words_selected = xl.parse('Primary_Word') # data frame

########################

# there are some 'nan' s in text in all articles
#all_articles = all_articles.reset_index().dropna().set_index('index')
#all_articles = all_articles.dropna().reset_index()
all_articles_selected = all_articles_selected.dropna().reset_index()
# getting companions frequencies of their primary words . It gives a dictionary
#linked_words = linking_words(p_words,all_articles,tokenizer)
linked_words_selected = linking_words(p_words_selected,all_articles_selected,tokenizer)
# saving the dict
#np.save('linked_words.npy', linked_words) 
np.save('linked_words_selected.npy', linked_words_selected) 
# Load
#read_dictionary = np.load('linked_words.npy').item()
#linked_words = np.load('linked_words.npy').item()
linked_words_selected = np.load('linked_words_selected.npy').item()
#
#get the most common ones
# also dict to dataframe
# 'collections counter' library to be used
from collections import Counter as cnt
common_companions = {}
keys = list(linked_words_selected.keys())
for key in keys:
    counted = cnt(linked_words_selected[key])
    commons = counted.most_common(10)
    t={}
    for k,v in commons:
        t[k] = v
    common_companions[key] = t


# to dataframe
commonCompanions = pd.DataFrame(common_companions) # columns are primary words
# get rid of nans
commonCompanions = commonCompanions.fillna(0)
# columns to rows
commonCompanions = commonCompanions.transpose()



"""
#################################
#################################
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
# here the max feature is to filter the less repeated words
cv = CountVectorizer() # you can check the parameters, there are some useful ones
# This below gets 0,1,2 and so
#X = cv.fit_transform(corpus).toarray() 
# So i wanna keep column headers as word to crawl them later; so that i transfrm t to dataframe
X = pd.DataFrame(cv.fit_transform(commonCompanions).toarray(), columns=cv.get_feature_names())
#y = dataset.iloc[0, 4].values

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
nX = sc_X.fit_transform(X)
#

"""
#######################
# NOT REALLY BUILT YET #
# STILL WORKING ON IT #


"""
# PCA
from sklearn.decomposition import PCA
# you put here how many dimension you want
# BUT you should check how luch of the variance is explained and so.
#That's why we don't just start by 2. So you put None to see the explaining percentage
#pca=PCA(n_components = None)
pca=PCA(n_components = 3)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

"""
# KMEANS
########

# scaling
X1 = commonCompanions
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
nX = sc_X.fit_transform(X1)

from sklearn.cluster import KMeans
# we create a loop to have 10 clusters stats to compare each
wcss = []
for i in range(1,11):
    # we use the kmeans ++ initialization method
    # cause rndom centroids may cause problems
    kmeans= KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(nX)
    wcss.append(kmeans.inertia_) # calculate the wcss distance stuff to see the most efficient cluster number
    
# now we plot the wcss
plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Wcss value")
plt.show()

# applying the kmeans on our dataset now.
kmeans= KMeans(n_clusters=4, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
# giving clusters to indiividuals
y_kmeans = kmeans.fit_predict(nX)

# Visualize the clusters
# we giving the coodinates with 0 and 1. ==0 is the cluster
# s = size , c= color
plt.scatter(nX[y_kmeans == 0, 0], nX[y_kmeans == 0, 1], s=100, c= 'red', label='Cluster 1') 
plt.scatter(nX[y_kmeans == 1, 0], nX[y_kmeans == 1, 1], s=100, c= 'blue', label='Cluster 2') 
plt.scatter(nX[y_kmeans == 2, 0], nX[y_kmeans == 2, 1], s=100, c= 'green', label='Cluster 3') 
plt.scatter(nX[y_kmeans == 3, 0], nX[y_kmeans == 3, 1], s=100, c= 'cyan', label='Cluster 4') 

# centroids
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c= 'yellow', label='Centroids') 
plt.title('Clusters of Primary Words')
plt.xlabel('x')
plt.ylabel('Groups')
plt.legend() # we want to add all the different scores
plt.show()
#
# Without scaling
#
# to numpy array for the graph
X2 = X1.values

wcss = []
for i in range(1,11):
    # we use the kmeans ++ initialization method
    # cause rndom centroids may cause problems
    kmeans= KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_) # calculate the wcss distance stuff to see the most efficient cluster number
    
# now we plot the wcss
plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Wcss value")
plt.show()

# applying the kmeans on our dataset now.
kmeans= KMeans(n_clusters=4, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
# giving clusters to indiividuals
y_kmeans = kmeans.fit_predict(X2)

# Visualize the clusters
# we giving the coodinates with 0 and 1. ==0 is the cluster
# s = size , c= color
plt.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1], s=100, c= 'red', label='Cluster 1') 
plt.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1], s=100, c= 'blue', label='Cluster 2') 
plt.scatter(X2[y_kmeans == 2, 0], X2[y_kmeans == 2, 1], s=100, c= 'green', label='Cluster 3') 
plt.scatter(X2[y_kmeans == 3, 0], X2[y_kmeans == 3, 1], s=100, c= 'cyan', label='Cluster 4') 
plt.scatter(X2[y_kmeans == 4, 0], X2[y_kmeans == 4, 1], s=100, c= 'yellow', label='Cluster 5') 

# centroids
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c= 'yellow', label='Centroids') 
plt.title('Clusters of Primary Words')
plt.xlabel('x')
plt.ylabel('Groups')
plt.legend() # we want to add all the different scores
plt.show()

#6 i want to plot these in 3d with pca... we have 32 dimension normally
# PCA
from sklearn.decomposition import PCA
# you put here how many dimension you want
# BUT you should check how luch of the variance is explained and so.
#That's why we don't just start by 2. So you put None to see the explaining percentage
#pca=PCA(n_components = None)
pca=PCA(n_components = 3)
X_pca = pca.fit_transform(X1)
explained_variance = pca.explained_variance_ratio_

wcss = []
for i in range(1,11):
    # we use the kmeans ++ initialization method
    # cause rndom centroids may cause problems
    kmeans= KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_) # calculate the wcss distance stuff to see the most efficient cluster number
    
# now we plot the wcss
plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Wcss value")
plt.show()

kmeans= KMeans(n_clusters=4, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
# giving clusters to indiividuals
y_kmeans = kmeans.fit_predict(X_pca)

#3d plotting
from __future__ import division
import matplotlib.pyplot as plt1
from mpl_toolkits.mplot3d import Axes3D

fig = plt1.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
z_offset = 3

# Plotting 3D points
#ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],'ro', alpha=0.6, color = ['g','b','r','c','y'], label='Cluster Plot')

ax.scatter(X_pca[y_kmeans == 0, 0], X_pca[y_kmeans == 0, 1], X_pca[y_kmeans == 0, 2], alpha = 0.9,  color= 'red', label='Cluster 1')
ax.scatter(X_pca[y_kmeans == 1, 0], X_pca[y_kmeans == 1, 1], X_pca[y_kmeans == 1, 2], alpha = 0.9, color= 'blue', label='Cluster 2')
ax.scatter(X_pca[y_kmeans == 2, 0], X_pca[y_kmeans == 2, 1], X_pca[y_kmeans == 2, 2], alpha = 0.9, color= 'green', label='Cluster 3')
ax.scatter(X_pca[y_kmeans == 3, 0], X_pca[y_kmeans == 3, 1], X_pca[y_kmeans == 3, 2], alpha = 0.9, color= 'cyan', label='Cluster 4')
#ax.scatter(X_pca[y_kmeans == 4, 0], X_pca[y_kmeans == 4, 1], X_pca[y_kmeans == 4, 2], alpha = 0.9, color= 'yellow', label='Cluster 5')

# write to a file the clustered primary words
primaries = X1.index.values
clustered = []
for i in range(0,len(primaries)):
    t = []
    t.append(primaries[i])
    t.append(y_kmeans[i])
    clustered.append(t)
    
arranged = []
for i in range(0,5):
    for c in clustered:
        if c[1] == i:
            arranged.append(c)
    
from pandas import ExcelWriter
writer = ExcelWriter('Clustered_P_Words_selected.xlsx')
df = pd.DataFrame(arranged, columns = ['P_word','Cluster'])
df.to_excel(writer, 'Clusters') 
writer.save()  

p = purity(arranged,4,groups,p_words_selected)
print(p)






#################
#My sphere trials (doesn't work)
u1 = np.linspace(kmeans.cluster_centers_[0,0], 2 * np.pi, 100)
v1 = np.linspace(kmeans.cluster_centers_[0,1], np.pi, 100)


x_sphere_1 = 1   * np.outer(np.cos(u1), np.sin(v1)) + kmeans.cluster_centers_[0,0]
y_sphere_1 = 0.5 * np.outer(np.sin(u1), np.sin(v1)) + kmeans.cluster_centers_[0,1]
z_sphere_1 = 1.5 * np.outer(np.ones(np.size(u1)), np.cos(v1)) + kmeans.cluster_centers_[0,2]
ax.plot_surface(x_sphere_1, y_sphere_1, z_sphere_1,
                rstride=10, cstride=10, linewidth=0.1, color='b', alpha=0.1)

# Sphere surface #2
# ------------------
u2 = np.linspace(0, 2 * np.pi, 100)
v2 = np.linspace(0, np.pi, 100)

x_sphere_2 = 1.5 * np.outer(np.cos(u2), np.sin(v2)) + kmeans.cluster_centers_[1,0]
y_sphere_2 = 1   * np.outer(np.sin(u2), np.sin(v2)) + kmeans.cluster_centers_[1,1]
z_sphere_2 = 1.8 * np.outer(np.ones(np.size(u2)), np.cos(v2)) + kmeans.cluster_centers_[1,2]
ax.plot_surface(x_sphere_2, y_sphere_2, z_sphere_2,
                rstride=10, cstride=10, linewidth=0.1, color='r', alpha=0.1)

# Sphere surface #3
# -----------------
u3 = np.linspace(0, 2 * np.pi, 100)
v3 = np.linspace(0, np.pi, 100)

x_sphere_3 = 1.5 * np.outer(np.cos(u3), np.sin(v3)) + kmeans.cluster_centers_[2,0]
y_sphere_3 = 1   * np.outer(np.sin(u3), np.sin(v3)) + kmeans.cluster_centers_[2,1]
z_sphere_3 = 2   * np.outer(np.ones(np.size(u3)), np.cos(v3)) + kmeans.cluster_centers_[2,2]
ax.plot_surface(x_sphere_3, y_sphere_3, z_sphere_3,
                rstride=10, cstride=10, linewidth=0.1, color='g', alpha=0.1)


u4 = np.linspace(0, 2 * np.pi, 100)
v4 = np.linspace(0, np.pi, 100)
x_sphere_4 = 1.5 * np.outer(np.cos(u4), np.sin(v4)) + kmeans.cluster_centers_[3,0]
y_sphere_4 = 1   * np.outer(np.sin(u4), np.sin(v4)) + kmeans.cluster_centers_[3,1]
z_sphere_4 = 2   * np.outer(np.ones(np.size(u4)), np.cos(v4)) + kmeans.cluster_centers_[3,2]
ax.plot_surface(x_sphere_4, y_sphere_4, z_sphere_4,
                rstride=10, cstride=10, linewidth=0.1, color='cyan', alpha=0.1)

u5 = np.linspace(0, 2 * np.pi, 100)
v5 = np.linspace(0, np.pi, 100)
x_sphere_5 = 1.5 * np.outer(np.cos(u5), np.sin(v5)) + kmeans.cluster_centers_[4,0]
y_sphere_5 = 1   * np.outer(np.sin(u5), np.sin(v5)) + kmeans.cluster_centers_[4,1]
z_sphere_5 = 2   * np.outer(np.ones(np.size(u5)), np.cos(v5)) + kmeans.cluster_centers_[4,2]
ax.plot_surface(x_sphere_5, y_sphere_5, z_sphere_5,
                rstride=10, cstride=10, linewidth=0.1, color='yellow', alpha=0.1)


plt1.show()




















