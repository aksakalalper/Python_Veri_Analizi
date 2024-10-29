import scipy
from scipy import spatial
from scipy.spatial import distance
import numpy

def oklidUzaklıgı():
    #ikisi arasındaki uzaklığı vurgular.
    #p=1 manhattan,p=2 öklid,p=3+ chebysew uzaklığı
        #deger ne kadar buyukse o kadar benzemez
    hasan=[30,4000]
    huseyin=[28,3500]
    mahmut=[59,12000]

    hhOkl=distance.minkowski(hasan,huseyin,p=1) 
    hmOkl=distance.minkowski(hasan,mahmut,p=1)
    mhOkl=distance.minkowski(mahmut,huseyin,p=1)

    print(hhOkl,hmOkl,mhOkl)
def cosinusBenzerligi():
#iki vektorun benzerligini ortaya cikarir.
    aVektoru=(3,4)
    bVektoru=(4,2)
    ABUzaklik=distance.cosine(aVektoru,bVektoru)
    ABBenzerlik=1-ABUzaklik
    print(f"benzerlik orani {ABBenzerlik}")
def jaccardUzakligi():  
    '''
isim	ders1	ders2	ders3	ders4
hasan	1	    1   	1	    0
huseyin	1	    0	    1	    0
ayse	0	    1	    0	    1
    '''
    hasan=(1,1,1,0)
    huseyin=(1,0,1,0)
    ayse=(0,1,0,1)
    hasanHuseyin=(1-distance.jaccard(hasan,huseyin))
    hasanAyse=(1-distance.jaccard(hasan,ayse))
    huseyinAyse=(1-distance.jaccard(huseyin,ayse))

    print(f"benzerlik durumlari {hasanHuseyin}, {hasanAyse}, {huseyinAyse}")

oklidUzaklıgı()
cosinusBenzerligi()
jaccardUzakligi()
