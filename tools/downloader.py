# Made by Bjoernar Remmen
# Short is better
import urllib2
import os.path as osp
import os
from parallel_sync import wget

file_path = os.path.dirname(__file__)
print file_path
root = osp.join(file_path,"..")
print root
glove_dir = osp.join(root,"data","lm_data","glove")
if not os.path.exists(glove_dir):
    os.makedirs(glove_dir)
word2vec_dir = osp.join(root,"data", "lm_data", "word2vec")
if not os.path.exists(word2vec_dir):
    os.makedirs(word2vec_dir)
print glove_dir
# glove [50D,150D,300D]

glove_urls = ["https://www.dropbox.com/s/cgy4mdstgpmtnw8/vectors_50.txt?dl=1", "https://www.dropbox.com/s/hpjosjxrgbnsyqs/vectors.txt?dl=1", "https://www.dropbox.com/s/8r4524eaus1irdp/vectors.txt?dl=1"]
w2v_50 = ["https://www.dropbox.com/s/bngbeepc3ec3r8h/w2v_wiki.model?dl=1", "https://www.dropbox.com/s/1mx2gcv315cej24/w2v_wiki.model.syn1neg.npy?dl=1","https://www.dropbox.com/s/w5si5ihg73zrbvz/w2v_wiki.model.wv.syn0.npy?dl=1"]
w2v_150 = ["https://www.dropbox.com/s/fh8ng75kehog3jh/w2v_wiki.model?dl=1", "https://www.dropbox.com/s/ddeji4x4t6ds3x1/w2v_wiki.model.syn1neg.npy?dl=1", "https://www.dropbox.com/s/ydmemky269fuoeh/w2v_wiki.model.wv.syn0.npy?dl=1"]
w2v_300 = ["https://www.dropbox.com/s/hhaw2zj81bvuatn/w2v_wiki.model?dl=1", "https://www.dropbox.com/s/2lwafcxqvi2xlbv/w2v_wiki.model.syn1neg.npy?dl=1", "https://www.dropbox.com/s/vpmqrm8q78en38h/w2v_wiki.model.wv.syn0.npy?dl=1"]

w2v_names_50 = ["w2v_wiki_50.model","w2v_wiki_50.model.syn1neg.npy", "w2v_wiki_50.model.wv.syn0.npy"]
w2v_names_150 = ["w2v_wiki_150.model","w2v_wiki_150.model.syn1neg.npy", "w2v_wiki_150.model.wv.syn0.npy"]
w2v_names_300 = ["w2v_wiki_300.model","w2v_wiki_300.model.syn1neg.npy", "w2v_wiki_300.model.wv.syn0.npy"]



'''def download_file(name, url):
    f = urllib2.urlopen(url)
    if not(os.path.exists(glove_dir)):
        os.makedirs(glove_dir)

    with open(osp.join(glove_dir,name ), "wb") as code:
        code.write(f.read())
'''


if __name__ == "__main__":
    vector_size = [50,150,300]
    vector_names_glove = ["glove_wiki_50.txt", "glove_wiki_150.txt", "glove_wiki_300.txt"]
    vector_names_word2vec_50 = ["w2v_wiki_50.txt"]
    print "Starting download"
    wget.download(glove_dir, urls=glove_urls, filenames=vector_names_glove, parallelism=3)

  #  print "Downloaded", vector_names_glove
    #wget.download(word2vec_dir, urls=w2v_50, filenames=w2v_names_50, parallelism=3)
    #print "Downloaded", w2v_names_50
    #wget.download(word2vec_dir, urls=w2v_150, filenames=w2v_names_150, parallelism=3)
   # print "Downloaded", w2v_names_150
    #wget.download(word2vec_dir, urls=w2v_300, filenames=w2v_names_300, parallelism=3)
   # print "Downloaded", w2v_names_300
   # wget.download(glove_dir, urls=[glove_urls[0]],filenames=[vector_names[0]],parallelism=3)
  #  wget.download(glove_dir, urls=[glove_urls[1]],filenames=[vector_names[1]],parallelism=3)
   # wget.download(glove_dir, urls=[glove_urls[2]],filenames=[vector_names[2]],parallelism=3)



    '''

    This takes some time. The files are quite big.
    I suggest you try out a new hobby while it is downloading.


    The files are created at startup and will grow until the program stops.
    To monitor size: watch ls -sh file1 file2 file3ve
    vectors_50.txt : 629 mb
    vectors_150.txt : 1.9 gb
    vectors_300.txt : 3.62 gb


    Do not stop the program, if you do you could end up with a corrupt file
    '''