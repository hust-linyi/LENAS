import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.39409307, 'dep_conv'), (0.49018687, 'conv'), (0.2037214756011963, 'identity'), (0.37377119064331055, 'down_dil_conv')],up=[(0.62938005, 'dep_conv'), (0.5902223, 'se_conv'), (0.3727233648300171, 'none'), (0.7088075280189514, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


