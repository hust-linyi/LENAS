import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.24136606, 'conv'), (0.27837527, 'identity'), (0.22761361598968505, 'identity'), (0.24927610158920288, 'avg_pool')],up=[(0.3458971, 'se_conv'), (0.402917, 'se_conv'), (0.2850897550582886, 'identity'), (0.30542096495628357, 'up_dep_conv')])", 1)
pickle.dump(a,f)
f.close()


