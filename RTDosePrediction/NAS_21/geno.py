import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.3744606, 'dep_conv'), (0.3545911, 'dil_conv'), (0.2544734239578247, 'none'), (0.27939364314079285, 'down_dep_conv')],up=[(0.26731086, 'dep_conv'), (0.23303242, 'conv'), (0.28367712497711184, 'none'), (0.2293822318315506, 'up_dil_conv')])", 1)
pickle.dump(a,f)
f.close()


