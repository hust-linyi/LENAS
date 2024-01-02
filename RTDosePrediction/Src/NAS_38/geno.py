import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.27702215, 'conv'), (0.35063308, 'dil_conv'), (0.2732957363128662, 'identity'), (0.29425565600395204, 'down_dep_conv')],up=[(0.30246672, 'conv'), (0.33389643, 'dep_conv'), (0.28261146545410154, 'identity'), (0.3260557949542999, 'up_dil_conv')])", 1)
pickle.dump(a,f)
f.close()


