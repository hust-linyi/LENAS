import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.22053677, 'dep_conv'), (0.24442041, 'dil_conv'), (0.2806766748428345, 'identity'), (0.33223761320114137, 'max_pool')],up=[(0.3939924, 'se_conv'), (0.29584587, 'se_conv'), (0.30132489204406737, 'none'), (0.4963672161102295, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


