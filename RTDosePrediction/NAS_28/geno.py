import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.47672042, 'dep_conv'), (0.52048546, 'dil_conv'), (0.27550225257873534, 'identity'), (0.49870365858078003, 'down_dil_conv')],up=[(0.4639539, 'se_conv'), (0.41720048, 'identity'), (0.3200322389602661, 'none'), (0.7699423432350159, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


