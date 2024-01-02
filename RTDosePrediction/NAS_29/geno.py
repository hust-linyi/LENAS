import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.35017598, 'identity'), (0.24430475, 'dil_conv'), (0.21760184764862062, 'none'), (0.2582529366016388, 'down_dep_conv')],up=[(0.25242656, 'dep_conv'), (0.2862034, 'dep_conv'), (0.20971298217773438, 'none'), (0.2050437331199646, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


