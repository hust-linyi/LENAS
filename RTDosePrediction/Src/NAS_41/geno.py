import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.3752986, 'se_conv'), (0.54461366, 'dep_conv'), (0.31624114513397217, 'identity'), (0.32397390604019166, 'max_pool')],up=[(0.6655896, 'dil_conv'), (0.5995393, 'conv'), (0.308330774307251, 'none'), (0.5231748223304749, 'up_dep_conv')])", 1)
pickle.dump(a,f)
f.close()


