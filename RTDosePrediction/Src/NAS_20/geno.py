import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.24097343, 'dep_conv'), (0.35073784, 'conv'), (0.23278589248657228, 'identity'), (0.2707945883274078, 'down_dil_conv')],up=[(0.35987863, 'se_conv'), (0.3421218, 'se_conv'), (0.3077152013778687, 'none'), (0.5003914833068848, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


