import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.50403386, 'se_conv'), (0.5609513, 'dep_conv'), (0.29069616794586184, 'none'), (0.3319161057472229, 'down_dil_conv')],up=[(0.40083066, 'conv'), (0.37968868, 'se_conv'), (0.3067605495452881, 'none'), (0.6232038736343384, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


