import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.25163895, 'identity'), (0.34212065, 'se_conv'), (0.20680480003356932, 'identity'), (0.25948882699012754, 'down_dep_conv')],up=[(0.36453357, 'conv'), (0.30829552, 'identity'), (0.28816542625427244, 'identity'), (0.4138350486755371, 'up_dil_conv')])", 1)
pickle.dump(a,f)
f.close()


