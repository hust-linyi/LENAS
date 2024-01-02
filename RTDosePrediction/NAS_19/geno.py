import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.35784853, 'dil_conv'), (0.2891893, 'dil_conv'), (0.24484293460845946, 'identity'), (0.31562057733535764, 'down_dil_conv')],up=[(0.40971416, 'conv'), (0.3597016, 'conv'), (0.36520848274230955, 'none'), (0.23176975548267365, 'up_conv')])", 1)
pickle.dump(a,f)
f.close()


