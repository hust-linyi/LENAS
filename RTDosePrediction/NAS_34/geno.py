import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.37713182, 'se_conv'), (0.48605204, 'conv'), (0.21661531925201416, 'identity'), (0.4122839570045471, 'down_dil_conv')],up=[(0.36846194, 'se_conv'), (0.51094955, 'identity'), (0.36931092739105226, 'identity'), (0.6852478981018066, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


