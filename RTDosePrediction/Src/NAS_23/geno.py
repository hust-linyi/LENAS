import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.28788587, 'dep_conv'), (0.35280773, 'se_conv'), (0.2537153005599976, 'none'), (0.20000000596046447, 'down_dep_conv')],up=[(0.42042613, 'se_conv'), (0.5640495, 'dep_conv'), (0.3352819919586182, 'identity'), (0.5194765329360962, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()



