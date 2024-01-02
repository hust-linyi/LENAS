import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.2672496, 'dep_conv'), (0.347945, 'se_conv'), (0.2892997980117798, 'none'), (0.4387993454933167, 'down_conv')],up=[(0.40026957, 'se_conv'), (0.26158237, 'dep_conv'), (0.24145123958587647, 'none'), (0.2699286937713623, 'up_dep_conv')])", 1)
pickle.dump(a,f)
f.close()


