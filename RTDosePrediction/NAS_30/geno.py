import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.47939005, 'se_conv'), (0.34142658, 'dep_conv'), (0.346633768081665, 'none'), (0.5708033680915833, 'down_dep_conv')],up=[(0.45212704, 'dil_conv'), (0.37280336, 'se_conv'), (0.3259221315383911, 'identity'), (0.5545511841773987, 'up_dep_conv')])", 1)
pickle.dump(a,f)
f.close()


