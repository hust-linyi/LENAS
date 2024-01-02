import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.3699315, 'dep_conv'), (0.42127913, 'se_conv'), (0.3746939659118652, 'none'), (0.8690855741500855, 'down_dep_conv')],up=[(0.592877, 'dil_conv'), (0.7019428, 'dil_conv'), (0.35842444896698, 'none'), (0.44170787930488586, 'up_dil_conv')])", 1)
pickle.dump(a,f)
f.close()


