import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.2882657, 'se_conv'), (0.66447365, 'se_conv'), (0.34454610347747805, 'none'), (0.20000000596046447, 'max_pool')],up=[(0.6629312, 'se_conv'), (0.4427213, 'se_conv'), (0.3101553201675415, 'none'), (0.6669021248817444, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


