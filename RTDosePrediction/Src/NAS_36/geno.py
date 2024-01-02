import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.4119495, 'se_conv'), (0.5189161, 'conv'), (0.2125913143157959, 'none'), (0.3999804139137268, 'down_dil_conv')],up=[(0.6491584, 'se_conv'), (0.6049219, 'se_conv'), (0.3754725933074951, 'none'), (0.7261173129081726, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


