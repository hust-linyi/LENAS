import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.31277823, 'dep_conv'), (0.54509157, 'identity'), (0.35610189437866213, 'identity'), (0.4506379008293152, 'max_pool')],up=[(0.7026303, 'se_conv'), (0.5274003, 'se_conv'), (0.26008055210113523, 'identity'), (0.43727347254753113, 'up_conv')])", 1)
pickle.dump(a,f)
f.close()


